import os
import json
import argparse
from train_arc_hyper import train_single_task_hypernetwork
from arc_primary_net import ARCPrimaryNetwork
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
    print("=" * 60)
    print("META-TRAINING HYPERNETWORK (MULTI-EPOCH, MultiStepLR)")
    print(f"Split: {split}, Epochs: {n_epochs}, Iters/task: {n_iterations_per_task}")
    print(f"Embedding dim: {emb_dim}, LR: {lr} (single LR, matching train_hyper.py)")
    print("=" * 60)

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

    print(f"Training on {len(task_names)} tasks from {split} split")

    # Initialize primary network (hypernetwork + embeddings, matching train_hyper.py pattern)
    checkpoint_path = os.path.join(output_dir, 'hypernetworks_arc.pth')
    start_epoch = 0

    net = ARCPrimaryNetwork(emb_dim=emb_dim).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
        print(f"Loaded {len(net)} existing task embeddings")
    else:
        print("Initializing new primary network (hypernetwork + embeddings)")

    print(f"ARCPrimaryNetwork: emb_dim={emb_dim}, n_tasks={len(net)}")

    # Store optimizer and scheduler per task (persist across epochs)
    task_optimizers = {}
    task_schedulers = {}

    # Multi-epoch training loop
    solutions = {}
    for epoch in range(start_epoch, n_epochs):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch + 1}/{n_epochs}")
        print("=" * 80)

        # Train on each task
        for task_idx, task_name in enumerate(task_names):
            print(f"\n[Epoch {epoch + 1}/{n_epochs}] Task {task_idx + 1}/{len(task_names)}: {task_name}")

            try:
                # Get or create optimizer/scheduler for this task (persist across epochs)
                task_optimizer = task_optimizers.get(task_name, None)
                task_scheduler = task_schedulers.get(task_name, None)

                # Short optimization per task (100 iters per epoch)
                # Pass existing optimizer + scheduler to continue across epochs
                # Note: For first epoch, scheduler (MultiStepLR) will be created
                total_iterations_for_task = n_epochs * n_iterations_per_task
                solution, net, task_optimizer, task_scheduler = train_single_task_hypernetwork(
                    task_name=task_name,
                    split=split,
                    n_iterations=n_iterations_per_task,  # 100 iters this epoch
                    total_iterations=total_iterations_for_task,  # 5000 total for MultiStepLR
                    emb_dim=emb_dim,
                    lr=lr,  # Single LR for both embedding and hypernetwork
                    freeze_hypernetwork=False,  # Train both
                    net=net,  # Pass ARCPrimaryNetwork (updated in-place!)
                    net_path=None,
                    optimizer=task_optimizer,  # Pass existing optimizer (persists momentum!)
                    scheduler=task_scheduler,  # Pass existing scheduler (persists LR schedule!)
                    save_net=False,  # We'll save manually
                    output_dir=output_dir
                )

                # Note: net is updated in-place during training
                # Note: task embedding is already in net.task_embeddings[task_name]

                # Store optimizer and scheduler for next epoch
                task_optimizers[task_name] = task_optimizer
                task_schedulers[task_name] = task_scheduler

                # Keep latest solution
                solutions[task_name] = solution

            except Exception as e:
                print(f"Error training task {task_name}: {e}")
                continue

        # Save checkpoint after each epoch (matching train_hyper.py pattern)
        if (epoch + 1) % save_interval == 0:
            print(f"\n{'=' * 60}")
            print(f"Saving checkpoint at epoch {epoch + 1}")
            torch.save({
                'net': net.state_dict(),
                'emb_dim': emb_dim,
                'epoch': epoch + 1,
                'n_epochs': n_epochs,
                'n_tasks': len(net)
            }, checkpoint_path)
            print(f"Primary network saved to {checkpoint_path}")
            print(f"Saved {len(net)} task embeddings")
            print("=" * 60)

    # Final save
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    torch.save({
        'net': net.state_dict(),
        'emb_dim': emb_dim,
        'epoch': n_epochs,
        'n_epochs': n_epochs,
        'n_tasks': len(net)
    }, checkpoint_path)
    print(f"Final primary network saved to {checkpoint_path}")
    print(f"Hypernetwork + {len(net)} task embeddings saved")

    # Save all solutions
    solutions_path = os.path.join(output_dir, 'all_solutions.json')
    with open(solutions_path, 'w') as f:
        json.dump(solutions, f, indent=2)
    print(f"All solutions saved to {solutions_path}")

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

    # Meta-train on training split
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
