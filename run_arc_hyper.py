import os
import json
import argparse
from train_arc_hyper import train_single_task_hypernetwork
from embedding_manager import EmbeddingManager
from hypernetwork_arc import HyperNetworkARC
import torch


def meta_train_hypernetwork(
    split='training',
    task_names=None,
    max_tasks=None,
    n_iterations_per_task=2000,
    emb_dim=128,
    lr_emb=0.02,
    lr_hyper=0.001,
    output_dir='./hypernetwork_outputs',
    save_interval=10
):
    """
    Meta-train a shared hypernetwork across multiple ARC-AGI tasks.

    Training strategy:
        1. Initialize shared hypernetwork (random or from checkpoint)
        2. For each task:
            a. Initialize/load task-specific embedding
            b. Train embedding + hypernetwork on this task
            c. Save embedding and updated hypernetwork
        3. Hypernetwork learns to generate good weights from embeddings

    Args:
        split: Dataset split to use
        task_names: List of task names (if None, loads all from split)
        max_tasks: Maximum number of tasks to train (None = all)
        n_iterations_per_task: Training iterations per task
        emb_dim: Puzzle embedding dimension
        lr_emb: Learning rate for embeddings
        lr_hyper: Learning rate for hypernetwork
        output_dir: Directory for outputs
        save_interval: Save hypernetwork every N tasks

    Returns:
        hypernetwork: Trained hypernetwork
        embedding_manager: Manager with all embeddings
        solutions: Dict mapping task_name -> solution
    """
    print("=" * 60)
    print("META-TRAINING HYPERNETWORK")
    print(f"Split: {split}, Embedding dim: {emb_dim}")
    print(f"LR_emb: {lr_emb}, LR_hyper: {lr_hyper}")
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

    # Initialize hypernetwork
    hypernetwork_path = os.path.join(output_dir, 'hypernetwork_shared.pth')
    if os.path.exists(hypernetwork_path):
        print(f"Loading hypernetwork from {hypernetwork_path}")
        checkpoint = torch.load(hypernetwork_path)
        hypernetwork = HyperNetworkARC(emb_dim=emb_dim).to(device)
        hypernetwork.load_state_dict(checkpoint['hypernetwork'])
        start_task_idx = checkpoint.get('task_idx', 0)
    else:
        print("Initializing new hypernetwork")
        hypernetwork = HyperNetworkARC(emb_dim=emb_dim).to(device)
        start_task_idx = 0

    # Initialize embedding manager
    embedding_dir = os.path.join(output_dir, 'embeddings')
    embedding_manager = EmbeddingManager(emb_dim=emb_dim, save_dir=embedding_dir, device=device)

    # Try to load existing embeddings
    if os.path.exists(embedding_dir):
        embedding_manager.load_all_embeddings()

    # Train on each task
    solutions = {}
    for task_idx, task_name in enumerate(task_names[start_task_idx:], start=start_task_idx):
        print("\n" + "=" * 60)
        print(f"Task {task_idx + 1}/{len(task_names)}: {task_name}")
        print("=" * 60)

        try:
            # Train this task
            solution, updated_hypernetwork, puzzle_emb = train_single_task_hypernetwork(
                task_name=task_name,
                split=split,
                n_iterations=n_iterations_per_task,
                emb_dim=emb_dim,
                lr_emb=lr_emb,
                lr_hyper=lr_hyper,
                hypernetwork_path=None,  # Use in-memory hypernetwork
                save_hypernetwork=False,  # We'll save manually
                save_embedding=False,  # We'll save via manager
                output_dir=output_dir
            )

            # Update hypernetwork (in-place)
            hypernetwork = updated_hypernetwork

            # Save embedding
            embedding_manager.embeddings[task_name] = puzzle_emb
            embedding_manager.save_embedding(task_name)

            # Save solution
            solutions[task_name] = solution

            # Periodic checkpoint
            if (task_idx + 1) % save_interval == 0:
                torch.save({
                    'hypernetwork': hypernetwork.state_dict(),
                    'emb_dim': emb_dim,
                    'task_idx': task_idx + 1,
                    'n_tasks_trained': task_idx + 1
                }, hypernetwork_path)
                print(f"Checkpoint saved at task {task_idx + 1}")

        except Exception as e:
            print(f"Error training task {task_name}: {e}")
            continue

    # Final save
    torch.save({
        'hypernetwork': hypernetwork.state_dict(),
        'emb_dim': emb_dim,
        'task_idx': len(task_names),
        'n_tasks_trained': len(task_names)
    }, hypernetwork_path)
    print(f"\nFinal hypernetwork saved to {hypernetwork_path}")

    embedding_manager.save_all_embeddings()
    embedding_manager.save_metadata()

    # Save all solutions
    solutions_path = os.path.join(output_dir, 'all_solutions.json')
    with open(solutions_path, 'w') as f:
        json.dump(solutions, f, indent=2)
    print(f"All solutions saved to {solutions_path}")

    return hypernetwork, embedding_manager, solutions


def inference_with_hypernetwork(
    task_name,
    split='evaluation',
    hypernetwork_path='./hypernetwork_outputs/hypernetwork_shared.pth',
    embedding_path=None,
    n_iterations=2000,
    output_dir='./hypernetwork_outputs'
):
    """
    Run inference on a single task using pre-trained hypernetwork.

    Two modes:
        1. With embedding: Use pre-trained embedding (fast, no training)
        2. Without embedding: Optimize new embedding (slower, still benefits from hypernetwork)

    Args:
        task_name: Task to solve
        split: Dataset split
        hypernetwork_path: Path to trained hypernetwork
        embedding_path: Path to embedding (None = optimize new one)
        n_iterations: Iterations if optimizing new embedding
        output_dir: Output directory

    Returns:
        solution: Task solution
    """
    print(f"Running inference on task: {task_name}")

    if embedding_path and os.path.exists(embedding_path):
        # Fast inference with pre-trained embedding
        print("Using pre-trained embedding (no optimization)")
        # TODO: Implement forward-only inference
        raise NotImplementedError("Fast inference mode not yet implemented")
    else:
        # Optimize embedding with frozen hypernetwork
        print("Optimizing new embedding with frozen hypernetwork")
        solution, _, _ = train_single_task_hypernetwork(
            task_name=task_name,
            split=split,
            n_iterations=n_iterations,
            lr_emb=0.02,
            lr_hyper=0.0,  # Freeze hypernetwork
            hypernetwork_path=hypernetwork_path,
            save_hypernetwork=False,
            output_dir=output_dir
        )
        return solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-train or use hypernetwork for ARC-AGI')
    parser.add_argument('--mode', type=str, choices=['meta_train', 'inference'], required=True)
    parser.add_argument('--split', type=str, default='training', choices=['training', 'evaluation', 'test'])
    parser.add_argument('--task', type=str, help='Task name (for inference mode)')
    parser.add_argument('--max_tasks', type=int, default=None, help='Max tasks to train (meta_train mode)')
    parser.add_argument('--iterations', type=int, default=2000, help='Iterations per task')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--lr_emb', type=float, default=0.02, help='Learning rate for embeddings')
    parser.add_argument('--lr_hyper', type=float, default=0.001, help='Learning rate for hypernetwork')
    parser.add_argument('--output_dir', type=str, default='./hypernetwork_outputs')
    parser.add_argument('--hypernetwork_path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=10, help='Save every N tasks')

    args = parser.parse_args()

    if args.mode == 'meta_train':
        meta_train_hypernetwork(
            split=args.split,
            max_tasks=args.max_tasks,
            n_iterations_per_task=args.iterations,
            emb_dim=args.emb_dim,
            lr_emb=args.lr_emb,
            lr_hyper=args.lr_hyper,
            output_dir=args.output_dir,
            save_interval=args.save_interval
        )
    elif args.mode == 'inference':
        if not args.task:
            raise ValueError("Must specify --task for inference mode")

        hypernetwork_path = args.hypernetwork_path or os.path.join(args.output_dir, 'hypernetwork_shared.pth')

        inference_with_hypernetwork(
            task_name=args.task,
            split=args.split,
            hypernetwork_path=hypernetwork_path,
            n_iterations=args.iterations,
            output_dir=args.output_dir
        )
