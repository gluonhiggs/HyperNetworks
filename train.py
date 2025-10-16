import numpy as np
import torch
from torch.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings
import matplotlib.pyplot as plt  # Add for plotting
from mpl_toolkits.mplot3d import Axes3D  # For 3D slice if needed
import torch.nn.functional as F

"""
This file trains a model for every ARC-AGI task in a split.
"""


def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.

    Args:
        mask: Tensor representing the mask
        length: Length of the slice to select

    Returns:
        log_partition: Log partition function
        logprobs: Log probabilities for each offset
    """
    if length > mask.shape[0]:
        return torch.tensor(float('-inf'), device=mask.device), torch.tensor([], dtype=torch.float, device=mask.device)

    cum_mask = torch.cat([torch.zeros(1, device=mask.device, dtype=mask.dtype), torch.cumsum(mask, dim=0)])
    offsets = torch.arange(mask.shape[0] - length + 1, device=mask.device)

    if offsets.numel() == 0:
        return torch.tensor(float('-inf'), device=mask.device), torch.tensor([], dtype=torch.float, device=mask.device)

    prefix = cum_mask[offsets]
    slice_sum = cum_mask[offsets + length] - cum_mask[offsets]
    suffix = cum_mask[-1] - cum_mask[offsets + length]
    logprobs = -prefix + slice_sum - suffix
    log_partition = torch.logsumexp(logprobs, dim=0) if logprobs.numel() > 0 else torch.tensor(float('-inf'), device=mask.device)

    return log_partition, logprobs
    
def take_step(task, model, optimizer, train_step, train_history_logger, total_steps, scheduler=None, scaler=None):
    """
    Runs a forward pass of the model on the ARC-AGI task.

    Note: This function contains the original ELBO calculation with mixed precision support.
    For a cleaner, reusable version of ELBO calculation, see compute_elbo in elbo_utils.py
    which is used for meta-learning in meta.py.

    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """

    optimizer.zero_grad()
    with autocast('cuda'):
        logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
        logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits
        

        # No beta annealing - use full KL weight for meta-training
        # (Beta annealing is designed for long training runs like 2000 steps,
        #  not needed for short 20-step meta-training iterations)
        beta = 1.0

        # Compute the total KL loss
        total_KL = beta * sum(torch.sum(kl) for kl in KL_amounts)

        # Vectorized reconstruction error: Batch over examples/modes
        reconstruction_error = 0.0
        for example_num in range(task.n_examples):
            for in_out_mode in range(2):
                if example_num >= task.n_train and in_out_mode == 1:
                    continue

                # Determine whether the grid size is already known.
                # If not, there is an extra term in the reconstruction error, corresponding to
                # the probability of reconstructing the correct grid size.
                # Check if grid size is known through any shape consistency
                shape_is_known = (task.in_out_same_size or 
                                 (task.all_out_same_size and in_out_mode==1) or 
                                 (task.all_in_same_size and in_out_mode==0)
                                 )
                grid_size_uncertain = not shape_is_known
                
                # Determine dimension-specific uncertainty for output mode
                if in_out_mode == 1:  # Output mode
                    # Check if X dimension is certain (all outputs have same X)
                    x_dimension_certain = (task.in_out_same_size or 
                                          task.all_out_same_size or 
                                          getattr(task, 'all_out_same_x', False))
                    # Check if Y dimension is certain (all outputs have same Y)
                    y_dimension_certain = (task.in_out_same_size or 
                                          task.all_out_same_size or 
                                          getattr(task, 'all_out_same_y', False))
                    
                    # Apply different coefficients for X and Y dimensions
                    if x_dimension_certain:
                        x_coefficient = 1.0
                    else:
                        x_coefficient = 0.01**max(0, 1-train_step/100)
                        
                    if y_dimension_certain:
                        y_coefficient = 1.0
                    else:
                        y_coefficient = 0.01**max(0, 1-train_step/100)
                else:  # Input mode - use uniform coefficient
                    x_coefficient = 1.0 if not grid_size_uncertain else 0.01**max(0, 1-train_step/100)
                    y_coefficient = x_coefficient
                logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
                problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
                output_shape = task.shapes[example_num][in_out_mode]
                
                x_log_partition, x_logprobs = mask_select_logprobs(x_coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
                y_log_partition, y_logprobs = mask_select_logprobs(y_coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
                # Account for probability of getting right grid size, if grid size is not known
                if grid_size_uncertain:
                    x_log_partitions = []
                    y_log_partitions = []
                    for length in range(1, x_mask.shape[1]+1):
                        x_log_partitions.append(mask_select_logprobs(x_coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                    for length in range(1, y_mask.shape[1]+1):
                        y_log_partitions.append(mask_select_logprobs(y_coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                    x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                    y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)
                
                # Given that we have the correct grid size, get the reconstruction error of getting the colors right
                logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
                for x_offset in range(x_logprobs.shape[0]):
                    for y_offset in range(y_logprobs.shape[0]):
                        logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition  # given the correct grid size,
                        logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                        target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y

                        logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')  # calculate the error for the colors.
                        logprobs[x_offset].append(logprob)
                logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
                # Use the maximum coefficient for aggregation to maintain proper scaling
                max_coefficient = max(x_coefficient, y_coefficient)
                logprob = torch.logsumexp(max_coefficient*logprobs, dim=(0,1))/max_coefficient  # Aggregate for all possible grid sizes
                reconstruction_error = reconstruction_error - logprob
            loss = total_KL + 10*reconstruction_error
    # Handle combinations of scaler (AMP) and scheduler (LR annealing)
    if scaler:
        # Use AMP (rare - we disable this in train_arc_hyper.py)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
    else:
        # No AMP (our case)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss
                             )
