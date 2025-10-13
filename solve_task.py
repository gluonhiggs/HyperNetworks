import os
import time
import json
import gc
import traceback
import sys

import torch
from torch.amp import GradScaler

import pre_processing
import train
from train import plot_elbo_progress
import arc_compressor
import solution_selection
import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings
import math
import numpy as np
import random

"""
A script that solves one puzzle, to be imported and used with parallel_train.py and multiprocessing.
"""
def multi_interval_lambda(t, T1, T2, T3, initial_lr, eta1, eta2):
    if t < T1:
        return eta1 / initial_lr + 0.5 * (1 - eta1 / initial_lr) * (1 + math.cos(math.pi * t / T1))
    elif t < T2:
        return eta1 / initial_lr  # Hold at eta1 ratio
    elif t < T3:
        return eta2 / initial_lr + 0.5 * ((eta1 - eta2) / initial_lr) * (1 + math.cos(math.pi * (t - T2) / (T3 - T2)))
    else:
        return eta2 / initial_lr
    
def cosine_hold_lambda(t, T_max, initial_lr, eta_min):
    if t < T_max:
        return (eta_min / initial_lr) + 0.5 * (1 - eta_min / initial_lr) * (1 + math.cos(math.pi * t / T_max))
    else:
        return eta_min / initial_lr  # Hold at eta_min ratio
    
def solve_task(task_name, split, time_limit, n_train_iterations, gpu_id, memory_dict, solutions_dict, error_queue):
    """
    Solves a puzzle.
    Args:
        task_name (str): The name of the puzzle to solve.
        split (str): 'training', 'evaluation', or 'test'
        time_limit (float): An end time that will cause training to exit early if reached.
        n_train_iterations (int): The number of iterations to train for.
        gpu_id (int): The GPU number to run the solver on.
        memory_dict (multiprocessing.Dict[str, int]): An inter-process shared dict that we
            can store the amount of memory taken by this job in.
        solutions_dict (multiprocessing.Dict[str, list[Dict[str, list[list[int]]]]]): An
            inter-process shared dict that we can store the solution in.
        error_queue (multiprocessing.Queue[Exception]): An inter-process shared queue to
            put errors in when an exception occurs.
    """

    try:  # Error catching block that puts errors on the error_queue

        seed = hash(task_name) % (2**32)  # Task-specific for unique but reproducible randomness
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats()  # Measure the memory used.

        # Get the task
        data_dir = os.environ.get('ARC_DATA_DIR')
        print("=" * 60)
        print(f"Solving tasks in {data_dir} for split '{split}'")
        with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
        task = pre_processing.Task(task_name, problems[task_name], None)
        del problems

        # Set up the training
        model = arc_compressor.ARCCompressor(task)
        base_lr = 0.02
        weight_decay = 1e-3
        T_max = 700
        eta_min= 1e-4
        optimizer = torch.optim.AdamW(model.weights_list, lr=base_lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(model.weights_list, lr=base_lr, betas=(0.5, 0.9), weight_decay=0.00020490192492053505)
        T_1 = 700
        T_2 = 1000
        T_0 = n_train_iterations//4 if n_train_iterations > 100 else 1
        train_history_logger = solution_selection.Logger(task)
        # train_history_logger.solution_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))
        # train_history_logger.solution_second_most_frequent = tuple(((0, 0), (0, 0)) for example_num in range(task.n_test))
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        # lr_lambda=lambda t: multi_interval_lambda(t, T_1, T_2, n_train_iterations, base_lr, 0.008, 0.001))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_train_iterations, eta_min=eta_min)  # Add scheduler
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #     lr_lambda=lambda t: cosine_hold_lambda(t, T_max, base_lr, eta_min))
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=T_0
        # )
        # scheduler = None
        scaler = GradScaler()  # Add AMP scaler
        # recon_history = [] # For early stopping

        # Training loop
        prev_loss = float('inf')
        patience = 7
        for train_step in range(n_train_iterations):
            train.take_step(task, model, optimizer, train_step, train_history_logger, n_train_iterations, scheduler, scaler)
            # train.take_step(task, model, optimizer, train_step, train_history_logger, n_train_iterations)
            if train_step % 50 == 0:  # Minimal monitoring: Print every 100 steps
                if train_step % 500 == 0:  
                    torch.cuda.empty_cache()
                # Loss is accessed from train_history_logger.loss_curve, updated by take_step
                loss = train_history_logger.loss_curve[-1] if train_history_logger.loss_curve else 'N/A'
                # if abs(loss - prev_loss) < 100 and loss <= 100:  # Adjusted threshold based on logs (~40-100 range, deltas ~1-10)
                #     patience -= 1
                #     if patience <= 0:
                #         print(f"Task {task_name} converged (plateau) around step {train_step}")
                #         break
                # else:
                #     patience = 7
                # prev_loss = loss
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Task {task_name} on GPU {gpu_id}: Step {train_step}/{n_train_iterations}, Loss: {loss}, Memory: {memory_used:.2f} GB")
                sys.stdout.flush()
            if time.time() > time_limit:
                break
        # plot_elbo_progress(train_history_logger)
        # Get the solution
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
        del task
        del model
        del optimizer
        del train_history_logger
        torch.cuda.empty_cache()
        gc.collect()

        # Store the result
        memory_dict[task_name] = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list

    except Exception as e:  # If error, write to the error queue
        error_queue.put(traceback.format_exc())
