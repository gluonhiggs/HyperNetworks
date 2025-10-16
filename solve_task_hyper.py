import os
import time
import json
import gc
import traceback
import sys

import torch
import torch.optim as optim
import numpy as np

import pre_processing
import train
import solution_selection
from primary_net_arc import ARCPrimaryNetwork
import warnings
warnings.filterwarnings("ignore")


def solve_task_hyper(task_name, split, time_limit, n_train_iterations, gpu_id,
                     memory_dict, solutions_dict, error_queue, checkpoint_path, emb_dim):
    """
    Solve ARC task using trained hypernetwork with test-time adaptation.

    Args:
        task_name: Task identifier
        split: 'training', 'evaluation', or 'test'
        time_limit: End time for early exit
        n_train_iterations: Number of optimization steps for embedding
        gpu_id: GPU device ID
        memory_dict: Shared dict for memory tracking
        solutions_dict: Shared dict for solutions
        error_queue: Shared queue for errors
        checkpoint_path: Path to trained hypernetwork checkpoint
        emb_dim: Embedding dimension
    """

    try:
        # Task-specific seeding for reproducibility
        seed = hash(task_name) % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.set_default_device('cuda')
        torch.cuda.set_device(gpu_id)
        torch.cuda.reset_peak_memory_stats()

        # Load task
        data_dir = os.environ.get('ARC_DATA_DIR')
        with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
            problems = json.load(f)
        task = pre_processing.Task(task_name, problems[task_name], None)
        del problems

        # Load trained hypernetwork (frozen)
        net = ARCPrimaryNetwork(emb_dim=emb_dim).cuda()
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        net.load_state_dict(checkpoint['net'])

        # Freeze hypernetwork - only optimize embedding
        net.hypernetwork.requires_grad_(False)
        if net.embedding_predictor:
            net.embedding_predictor.requires_grad_(False)

        # Create task-specific embedding (initialized from metadata predictor)
        puzzle_embedding = net.get_or_create_embedding(task_name, task)

        # Optimizer for embedding only
        lr = 0.01
        optimizer = optim.AdamW(puzzle_embedding.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=5e-4)

        # Model wrapper for train.take_step interface
        class ModelWrapper:
            def __init__(self, net, task, task_name):
                self.net = net
                self.task = task
                self.task_name = task_name
            def forward(self):
                return self.net(self.task, self.task_name)

        model = ModelWrapper(net, task, task_name)
        train_history_logger = solution_selection.Logger(task)

        # Test-time adaptation: optimize embedding
        for train_step in range(n_train_iterations):
            train.take_step(task, model, optimizer, train_step, train_history_logger,
                          n_train_iterations, None, None)

            if train_step % 100 == 0:
                if train_step % 500 == 0:
                    torch.cuda.empty_cache()
                loss = train_history_logger.loss_curve[-1] if train_history_logger.loss_curve else 'N/A'
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Task {task_name} on GPU {gpu_id}: Step {train_step}/{n_train_iterations}, Loss: {loss}, Memory: {memory_used:.2f} GB")
                sys.stdout.flush()

            if time.time() > time_limit:
                break

        # Extract solution
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})

        # Cleanup
        del task
        del net
        del optimizer
        del train_history_logger
        torch.cuda.empty_cache()
        gc.collect()

        # Store results
        memory_dict[task_name] = torch.cuda.max_memory_allocated()
        solutions_dict[task_name] = example_list

    except Exception as e:
        error_queue.put(traceback.format_exc())
