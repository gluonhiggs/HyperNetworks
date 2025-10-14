import os
import json
import torch
import torch.optim as optim
from torch.amp import GradScaler
import numpy as np

import pre_processing
import train
from arc_primary_net import ARCPrimaryNetwork
import solution_selection


def train_single_task_hypernetwork(
    task_name,
    split='training',
    n_iterations=2000,
    total_iterations=None,
    emb_dim=128,
    lr=0.01,
    freeze_hypernetwork=False,
    net=None,
    net_path=None,
    optimizer=None,
    scheduler=None,
    save_net=True,
    output_dir='./hypernetwork_outputs'
):
    """
    Train a single task using hypernetwork architecture (matching CIFAR-10 pattern).

    This function optimizes the ARCPrimaryNetwork end-to-end model, which contains:
        1. The puzzle embedding for this specific task
        2. The hypernetwork weights (if not frozen)

    Args:
        task_name: Name of the ARC-AGI task to solve
        split: Dataset split ('training', 'evaluation', or 'test')
        n_iterations: Number of training iterations for this call
        total_iterations: Total iterations across all epochs (for scheduler milestones, None = single-task mode)
        emb_dim: Dimension of puzzle embedding
        lr: Learning rate (single LR for both embedding and hypernetwork, matching train_hyper.py)
        freeze_hypernetwork: Whether to freeze hypernetwork (True for test-time inference)
        net: Pre-initialized ARCPrimaryNetwork (takes precedence over path)
        net_path: Path to load pre-trained ARCPrimaryNetwork (None = random init)
        optimizer: Pre-initialized optimizer (None = create new, persists across epochs)
        scheduler: Pre-initialized LR scheduler (None = create new, persists across epochs)
        save_net: Whether to save network after training
        output_dir: Directory to save outputs

    Returns:
        solution: Dictionary with attempt_1 and attempt_2 predictions
        net: Trained ARCPrimaryNetwork
        optimizer: The optimizer used (for persistence)
        scheduler: The scheduler used (for persistence)
    """
    print("=" * 60)
    print(f"Training task: {task_name}")
    print(f"Embedding dim: {emb_dim}, LR: {lr}, Freeze hypernet: {freeze_hypernetwork}")
    print("=" * 60)

    # Setup device
    torch.set_default_device('cuda')
    device = torch.device('cuda')

    # Set seed for reproducibility
    seed = hash(task_name) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load task
    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    task = pre_processing.Task(task_name, problems[task_name], None)
    del problems

    # Initialize ARCPrimaryNetwork (matches CIFAR-10 PrimaryNetwork pattern)
    if net is not None:
        print("Using provided ARCPrimaryNetwork instance")
        # net is already initialized, just use it
    elif net_path and os.path.exists(net_path):
        print(f"Loading ARCPrimaryNetwork from {net_path}")
        checkpoint = torch.load(net_path)
        net = ARCPrimaryNetwork(emb_dim=emb_dim).to(device)
        net.load_state_dict(checkpoint['net'])
    else:
        print("Initializing new ARCPrimaryNetwork")
        net = ARCPrimaryNetwork(emb_dim=emb_dim).to(device)

    # Setup optimizer and scheduler (create only if not provided)
    if optimizer is None:
        print("Creating new optimizer (Adam with single LR, matching train_hyper.py)")

        # Get task embedding to ensure it exists
        _ = net.get_or_create_embedding(task_name)

        if not freeze_hypernetwork:
            # Optimize all parameters (matching CIFAR-10)
            optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.9))
            print(f"Optimizing entire network (embedding + hypernetwork) with lr={lr}")
        else:
            # Optimize only task embedding
            params_to_optimize = list(net.task_embeddings[task_name].parameters())
            optimizer = optim.Adam(params_to_optimize, lr=lr, betas=(0.5, 0.9))
            print(f"Optimizing task embedding only (hypernetwork frozen), lr={lr}")
    else:
        print("Using provided optimizer (continuing optimization across epochs)")

    # MultiStepLR scheduler (matching train_hyper.py pattern)
    if scheduler is None and total_iterations is not None:
        # For multi-epoch training, use MultiStepLR like CIFAR-10
        # Scale milestones proportionally to total iterations
        # CIFAR-10: 1M iters, milestones at [168k, 336k, 400k, 450k, 550k, 600k]
        # ARC: ~2M iters (50 epochs × 400 tasks × 100 iters), scale accordingly
        milestones = [
            int(total_iterations * 0.17),  # ~17% through training
            int(total_iterations * 0.34),  # ~34%
            int(total_iterations * 0.40),  # ~40%
            int(total_iterations * 0.45),  # ~45%
            int(total_iterations * 0.55),  # ~55%
            int(total_iterations * 0.60),  # ~60%
        ]
        print(f"Creating MultiStepLR scheduler (milestones={milestones}, gamma=0.5)")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )
    elif scheduler is not None:
        print("Using provided scheduler (continuing schedule across epochs)")
    else:
        print("No LR scheduler (single-task mode)")
        scheduler = None

    # AMP scaler
    scaler = GradScaler()

    # Logger
    train_history_logger = solution_selection.Logger(task)

    # Create wrapper to adapt net.forward(task, task_name) to model.forward()
    # This is needed because train.take_step expects model.forward() with no args
    class ModelWrapper:
        def __init__(self, net, task, task_name):
            self.net = net
            self.task = task
            self.task_name = task_name

        def forward(self):
            return self.net(self.task, self.task_name)

    model = ModelWrapper(net, task, task_name)

    # Training loop (matching CIFAR-10 pattern but using existing train.take_step)
    print(f"Starting training for {n_iterations} iterations...")
    for train_step in range(n_iterations):
        # Use existing train.take_step (it calls model.forward() internally)
        train.take_step(
            task, model, optimizer, train_step,
            train_history_logger, n_iterations, scheduler, scaler
        )

        # Logging
        if train_step % 50 == 0:
            loss = train_history_logger.loss_curve[-1] if train_history_logger.loss_curve else 'N/A'
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Step {train_step}/{n_iterations}, Loss: {loss:.2f}, Memory: {memory_used:.2f} GB")

    print("Training complete!")

    # Extract solutions
    example_list = []
    for example_num in range(task.n_test):
        attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
        attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
        example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    if save_net:
        # Save like CIFAR-10: single checkpoint with net.state_dict()
        net_path = os.path.join(output_dir, f'arc_primary_net_{task_name}.pth')
        torch.save({
            'net': net.state_dict(),
            'emb_dim': emb_dim,
            'task_name': task_name
        }, net_path)
        print(f"Saved ARCPrimaryNetwork to {net_path}")

    # Save solution
    solution_path = os.path.join(output_dir, f'solution_{task_name}.json')
    with open(solution_path, 'w') as f:
        json.dump({task_name: example_list}, f, indent=2)
    print(f"Saved solution to {solution_path}")

    return example_list, net, optimizer, scheduler


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train single ARC task with hypernetwork')
    parser.add_argument('--task', type=str, required=True, help='Task name to train')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'evaluation', 'test'])
    parser.add_argument('--iterations', type=int, default=2000, help='Number of training iterations')
    parser.add_argument('--emb_dim', type=int, default=128, help='Puzzle embedding dimension')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (single LR for embedding + hypernetwork)')
    parser.add_argument('--freeze_hypernetwork', action='store_true', help='Freeze hypernetwork (test-time inference)')
    parser.add_argument('--net_path', type=str, default=None, help='Path to pre-trained ARCPrimaryNetwork')
    parser.add_argument('--output_dir', type=str, default='./hypernetwork_outputs', help='Output directory')

    args = parser.parse_args()

    train_single_task_hypernetwork(
        task_name=args.task,
        split=args.split,
        n_iterations=args.iterations,
        emb_dim=args.emb_dim,
        lr=args.lr,
        freeze_hypernetwork=args.freeze_hypernetwork,
        net_path=args.net_path,
        output_dir=args.output_dir
    )
