import os
import json
import torch
import torch.optim as optim
from torch.amp import GradScaler
import numpy as np

import pre_processing
import train
from arc_compressor_hyper import ARCCompressorHyper
from hypernetwork_arc import HyperNetworkARC
import solution_selection


def train_single_task_hypernetwork(
    task_name,
    split='training',
    n_iterations=2000,
    emb_dim=128,
    lr_emb=0.02,
    lr_hyper=0.001,
    hypernetwork_path=None,
    save_hypernetwork=True,
    save_embedding=True,
    output_dir='./hypernetwork_outputs'
):
    """
    Train a single task using hypernetwork architecture.

    This function optimizes both:
        1. The puzzle embedding for this specific task
        2. The hypernetwork weights (if not frozen)

    Args:
        task_name: Name of the ARC-AGI task to solve
        split: Dataset split ('training', 'evaluation', or 'test')
        n_iterations: Number of training iterations
        emb_dim: Dimension of puzzle embedding
        lr_emb: Learning rate for puzzle embedding
        lr_hyper: Learning rate for hypernetwork (set to 0 to freeze hypernetwork)
        hypernetwork_path: Path to load pre-trained hypernetwork (None = random init)
        save_hypernetwork: Whether to save hypernetwork after training
        save_embedding: Whether to save puzzle embedding after training
        output_dir: Directory to save outputs

    Returns:
        solution: Dictionary with attempt_1 and attempt_2 predictions
        hypernetwork: Trained hypernetwork
        puzzle_emb: Trained puzzle embedding
    """
    print("=" * 60)
    print(f"Training task: {task_name}")
    print(f"Embedding dim: {emb_dim}, LR_emb: {lr_emb}, LR_hyper: {lr_hyper}")
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

    # Initialize hypernetwork
    if hypernetwork_path and os.path.exists(hypernetwork_path):
        print(f"Loading hypernetwork from {hypernetwork_path}")
        checkpoint = torch.load(hypernetwork_path)
        hypernetwork = HyperNetworkARC(emb_dim=emb_dim).to(device)
        hypernetwork.load_state_dict(checkpoint['hypernetwork'])
    else:
        print("Initializing new hypernetwork")
        hypernetwork = HyperNetworkARC(emb_dim=emb_dim).to(device)

    # Initialize puzzle embedding (random init)
    puzzle_emb = torch.nn.Parameter(
        torch.randn(emb_dim, device=device) * 0.01
    )

    # Create model
    model = ARCCompressorHyper(task, hypernetwork, puzzle_emb)

    # Setup optimizers
    optimizer_params = []

    # Puzzle embedding (always optimized)
    optimizer_params.append({'params': [puzzle_emb], 'lr': lr_emb})

    # Hypernetwork (optional)
    if lr_hyper > 0:
        optimizer_params.append({'params': hypernetwork.parameters(), 'lr': lr_hyper})
        print(f"Optimizing hypernetwork with lr={lr_hyper}")
    else:
        print("Hypernetwork frozen (lr=0)")

    optimizer = optim.AdamW(optimizer_params, betas=(0.5, 0.9), weight_decay=1e-3)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=1e-4
    )

    # AMP scaler
    scaler = GradScaler()

    # Logger
    train_history_logger = solution_selection.Logger(task)

    # Training loop
    print(f"Starting training for {n_iterations} iterations...")
    for train_step in range(n_iterations):
        # Recreate model with updated embedding/weights
        model = ARCCompressorHyper(task, hypernetwork, puzzle_emb)

        # Take optimization step
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

    if save_hypernetwork:
        hyper_path = os.path.join(output_dir, f'hypernetwork_{task_name}.pth')
        torch.save({
            'hypernetwork': hypernetwork.state_dict(),
            'emb_dim': emb_dim,
            'task_name': task_name
        }, hyper_path)
        print(f"Saved hypernetwork to {hyper_path}")

    if save_embedding:
        emb_path = os.path.join(output_dir, f'embedding_{task_name}.pth')
        torch.save({
            'embedding': puzzle_emb.detach().cpu(),
            'task_name': task_name,
            'emb_dim': emb_dim
        }, emb_path)
        print(f"Saved embedding to {emb_path}")

    # Save solution
    solution_path = os.path.join(output_dir, f'solution_{task_name}.json')
    with open(solution_path, 'w') as f:
        json.dump({task_name: example_list}, f, indent=2)
    print(f"Saved solution to {solution_path}")

    return example_list, hypernetwork, puzzle_emb


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train single ARC task with hypernetwork')
    parser.add_argument('--task', type=str, required=True, help='Task name to train')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'evaluation', 'test'])
    parser.add_argument('--iterations', type=int, default=2000, help='Number of training iterations')
    parser.add_argument('--emb_dim', type=int, default=128, help='Puzzle embedding dimension')
    parser.add_argument('--lr_emb', type=float, default=0.02, help='Learning rate for embedding')
    parser.add_argument('--lr_hyper', type=float, default=0.001, help='Learning rate for hypernetwork (0=freeze)')
    parser.add_argument('--hypernetwork_path', type=str, default=None, help='Path to pre-trained hypernetwork')
    parser.add_argument('--output_dir', type=str, default='./hypernetwork_outputs', help='Output directory')

    args = parser.parse_args()

    train_single_task_hypernetwork(
        task_name=args.task,
        split=args.split,
        n_iterations=args.iterations,
        emb_dim=args.emb_dim,
        lr_emb=args.lr_emb,
        lr_hyper=args.lr_hyper,
        hypernetwork_path=args.hypernetwork_path,
        output_dir=args.output_dir
    )
