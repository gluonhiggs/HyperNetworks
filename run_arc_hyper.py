import os
import json
import argparse
from train_arc_hyper import train_single_task_hypernetwork
from primary_net_arc import ARCPrimaryNetwork
import torch


def meta_train_hypernetwork(
    split='training',
    task_names=None,
    max_tasks=None,
    n_epochs=50,
    n_iterations_per_task=100,
    emb_dim=128,
    lr=0.01,
    output_dir='./hypernetwork_outputs',
    save_interval=1
):
    """
    Meta-train a shared hypernetwork across multiple ARC-AGI tasks.

    Multi-epoch training strategy (avoids catastrophic forgetting):
        1. Initialize ARCPrimaryNetwork (hypernetwork + embeddings, random or from checkpoint)
        2. For each epoch:
            a. For each task:
                - Get/create task-specific embedding from net.task_embeddings
                - Short optimization (100 iters) on this task
                - Update both embedding and net.hypernetwork
            b. Save checkpoint after epoch (net.state_dict() like CIFAR-10)
        3. Hypernetwork learns cross-task weight generation patterns

    This is analogous to CIFAR-10 training but with key differences:
        - CIFAR: 1 batch (128 images) → 1 optimizer.step()
        - ARC: 1 task (compression problem) → 100 optimizer.steps()
        - Reason: Can't batch ARC tasks due to structural variability

    Args:
        split: Dataset split to use
        task_names: List of task names (if None, loads all from split)
        max_tasks: Maximum number of tasks to train (None = all)
        n_epochs: Number of epochs through all tasks (default 50)
        n_iterations_per_task: Training iterations per task per epoch (default 100)
        emb_dim: Puzzle embedding dimension
        lr: Learning rate (single LR for both embedding and hypernetwork, matching train_hyper.py)
        output_dir: Directory for outputs
        save_interval: Save checkpoint every N epochs (default 1)

    Returns:
        net: Trained ARCPrimaryNetwork (contains hypernetwork + embeddings)
        solutions: Dict mapping task_name -> solution
    """
    print(f"\n{'='*60}\nMETA-TRAINING HYPERNETWORK")
    print(f"Split: {split} | Epochs: {n_epochs} | Iters/task: {n_iterations_per_task}")
    print(f"Embedding: {emb_dim}D | LR: {lr}\n{'='*60}")

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda')

    # Load task list
    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)

    if task_names is None:
        task_names = list(problems.keys())

    if max_tasks is not None:
        task_names = task_names[:max_tasks]

    print(f"\nTraining on {len(task_names)} tasks from {split} split")

    # Initialize or load network
    checkpoint_path = os.path.join(output_dir, 'hypernetworks_arc.pth')
    net = ARCPrimaryNetwork(emb_dim=emb_dim).to(device)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch} | {len(net)} embeddings loaded")
    else:
        print("Initializing new network")

    # Create ONE shared optimizer for entire network (matching CIFAR-10 pattern)
    # This optimizer persists across ALL tasks and ALL epochs
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.9))

    # MultiStepLR scheduler with milestones scaled to total iterations
    # Total iterations = n_epochs × n_iterations_per_task × n_tasks
    total_iterations = n_epochs * n_iterations_per_task * len(task_names)
    milestones = [
        int(total_iterations * 0.17),
        int(total_iterations * 0.34),
        int(total_iterations * 0.40),
        int(total_iterations * 0.45),
        int(total_iterations * 0.55),
        int(total_iterations * 0.60),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    print(f"Shared optimizer created: total_iters={total_iterations}, milestones={milestones}")

    # Multi-epoch training loop
    solutions = {}
    for epoch in range(start_epoch, n_epochs):
        print(f"\n{'='*80}\nEPOCH {epoch+1}/{n_epochs}\n{'='*80}")

        for task_idx, task_name in enumerate(task_names):
            print(f"[Epoch {epoch+1}/{n_epochs}] Task {task_idx+1}/{len(task_names)}: {task_name}")

            try:
                # Train this task using SHARED optimizer/scheduler
                solution, net, _, _ = train_single_task_hypernetwork(
                    task_name=task_name,
                    split=split,
                    n_iterations=n_iterations_per_task,
                    total_iterations=None,  # Don't create new scheduler in train_single_task_hypernetwork
                    emb_dim=emb_dim,
                    lr=lr,
                    freeze_hypernetwork=False,
                    net=net,
                    net_path=None,
                    optimizer=optimizer,  # Shared optimizer
                    scheduler=scheduler,  # Shared scheduler
                    save_net=False,
                    output_dir=output_dir
                )

                solutions[task_name] = solution

            except Exception as e:
                print(f"ERROR: {task_name}: {e}")
                continue

        # Save checkpoint after each epoch
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'net': net.state_dict(),
                'emb_dim': emb_dim,
                'epoch': epoch + 1,
                'n_epochs': n_epochs,
                'n_tasks': len(net)
            }, checkpoint_path)
            print(f"Checkpoint saved: epoch {epoch+1} | {len(net)} embeddings")

    # Final save
    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")
    torch.save({
        'net': net.state_dict(),
        'emb_dim': emb_dim,
        'epoch': n_epochs,
        'n_epochs': n_epochs,
        'n_tasks': len(net)
    }, checkpoint_path)

    solutions_path = os.path.join(output_dir, 'all_solutions.json')
    with open(solutions_path, 'w') as f:
        json.dump(solutions, f, indent=2)

    print(f"Network saved: {checkpoint_path} | {len(net)} embeddings")
    print(f"Solutions saved: {solutions_path}")

    return net, solutions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-train hypernetwork for ARC-AGI')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Training configuration (hardcoded like train_hyper.py)
    emb_dim = 128
    learning_rate = 0.01
    n_epochs = 50
    n_iterations_per_task = 100
    output_dir = './hypernetwork_outputs'
    save_interval = 1

    print("=" * 60)
    print("ARC-AGI HYPERNETWORK TRAINING")
    print(f"Embedding dim: {emb_dim}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}, Iterations/task: {n_iterations_per_task}")
    print("=" * 60)

    # Meta-train on all training tasks
    meta_train_hypernetwork(
        split='training',  # Training data
        task_names=None,   # All tasks
        max_tasks=None,    # No limit
        n_epochs=n_epochs,
        n_iterations_per_task=n_iterations_per_task,
        emb_dim=emb_dim,
        lr=learning_rate,
        output_dir=output_dir,
        save_interval=save_interval
    )
