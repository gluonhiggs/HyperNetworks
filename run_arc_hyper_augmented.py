import os
import json
import random
import torch
import torch.optim as optim

from train_arc_hyper import train_single_task
from primary_net_arc import ARCPrimaryNetwork
import utils


def load_tasks_with_augmentation(split, augmentation_types=None):
    """
    Load tasks and create augmented versions.

    Args:
        split: 'training', 'evaluation', etc.
        augmentation_types: List of augmentation types to apply.
                          Default: ['original', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']

    Returns:
        List of augmented task names in format: 'task_id_augtype' or 'task_id' for original
    """
    if augmentation_types is None:
        augmentation_types = ['original', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']

    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)

    original_tasks = list(problems.keys())
    augmented_task_list = []

    for task_id in original_tasks:
        for aug_type in augmentation_types:
            if aug_type == 'original':
                augmented_task_list.append(task_id)
            else:
                augmented_task_list.append(f"{task_id}_{aug_type}")

    return augmented_task_list


def load_tasks(split):
    """Original load_tasks for non-augmented training."""
    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    return list(problems.keys())


def parse_augmented_task_name(augmented_name):
    """
    Parse augmented task name into original ID and augmentation type.

    Examples:
        'c9680e90' -> ('c9680e90', 'original')
        'c9680e90_rot90' -> ('c9680e90', 'rot90')
        'c9680e90_flip_h' -> ('c9680e90', 'flip_h')
    """
    if '_' not in augmented_name:
        return augmented_name, 'original'

    # Handle special cases with underscore in aug type (flip_h, flip_v)
    parts = augmented_name.split('_')
    if len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) == 3 and parts[1] == 'flip':
        return parts[0], f"{parts[1]}_{parts[2]}"
    else:
        # Fallback: treat everything after first underscore as aug type
        task_id = parts[0]
        aug_type = '_'.join(parts[1:])
        return task_id, aug_type


def apply_augmentation(task_dict, aug_type):
    """
    Apply augmentation to task dictionary.

    Args:
        task_dict: Original task with 'train' and 'test' keys
        aug_type: One of 'original', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v'

    Returns:
        Augmented task dictionary
    """
    if aug_type == 'original':
        return task_dict

    aug_mapping = {
        'rot90': ('rotate', 1),
        'rot180': ('rotate', 2),
        'rot270': ('rotate', 3),
        'flip_h': ('flip', 0),  # horizontal flip
        'flip_v': ('flip', 1),  # vertical flip
    }

    if aug_type not in aug_mapping:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

    aug_base, seed = aug_mapping[aug_type]
    return utils.augment_task(task_dict, augmentation_type=aug_base, seed=seed)


def train_single_task_augmented(augmented_task_name, split, n_iterations, emb_dim, lr,
                                net, optimizer, lr_scheduler=None, save_solution=False,
                                output_dir='./hypernetwork_outputs_augmented'):
    """
    Train single task with augmentation support.

    This wrapper handles augmented task names, applies augmentation, then calls
    the original train_single_task with the augmented task name for embedding purposes.
    """
    import pre_processing
    import torch
    import numpy as np
    import solution_selection
    import train as train_module

    # Parse augmented name
    original_task_id, aug_type = parse_augmented_task_name(augmented_task_name)

    # Set seed for reproducibility
    seed = hash(augmented_task_name) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load original task
    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)

    # Apply augmentation
    task_dict = apply_augmentation(problems[original_task_id], aug_type)
    task = pre_processing.Task(augmented_task_name, task_dict, None)
    del problems

    # Get/create embedding for augmented task (uses augmented_task_name)
    _ = net.get_or_create_embedding(augmented_task_name, task)

    # Setup logger
    if save_solution:
        train_history_logger = solution_selection.Logger(task)
    else:
        class LightweightLogger:
            def __init__(self):
                self.loss_curve = []
            def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss):
                self.loss_curve.append(float(loss.detach().cpu().numpy()))
        train_history_logger = LightweightLogger()

    # Model wrapper
    class ModelWrapper:
        def __init__(self, net, task, task_name):
            self.net = net
            self.task = task
            self.task_name = task_name
        def forward(self):
            return self.net(self.task, self.task_name)

    model = ModelWrapper(net, task, augmented_task_name)

    # Training loop
    for train_step in range(n_iterations):
        train_module.take_step(task, model, optimizer, train_step, train_history_logger,
                              n_iterations, lr_scheduler, None)

    # Save solution if requested
    if save_solution and hasattr(train_history_logger, 'solution_most_frequent'):
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})

        os.makedirs(output_dir, exist_ok=True)
        solution_path = os.path.join(output_dir, f'solution_{augmented_task_name}.json')
        with open(solution_path, 'w') as f:
            json.dump({augmented_task_name: example_list}, f, indent=2)

        return example_list

    return None


def validate(net, split, n_steps, max_tasks, emb_dim, lr):
    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')

    challenges_path = f'{data_dir}/arc-agi_{split}_challenges.json'
    solutions_path = f'{data_dir}/arc-agi_{split}_solutions.json'

    if not os.path.exists(challenges_path):
        alt_data_dir = 'dataset_old/'
        if os.path.exists(f'{alt_data_dir}/arc-agi_{split}_challenges.json'):
            data_dir = alt_data_dir
            challenges_path = f'{data_dir}/arc-agi_{split}_challenges.json'
            solutions_path = f'{data_dir}/arc-agi_{split}_solutions.json'

    with open(challenges_path, 'r') as f:
        problems = json.load(f)
    with open(solutions_path, 'r') as f:
        solutions_gt = json.load(f)

    task_names = list(problems.keys())[:max_tasks]

    net.hypernetwork.requires_grad_(False)
    if net.embedding_predictor:
        net.embedding_predictor.requires_grad_(False)

    n_correct_1 = 0
    n_correct_2 = 0
    n_total = 0

    for task_name in task_names:
        try:
            optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=5e-4)
            solution = train_single_task(task_name, split, n_steps, emb_dim, lr, net, optimizer, None, save_solution=True)

            gt = solutions_gt[task_name]

            solved_1 = all(solution[i]['attempt_1'] == gt[i] for i in range(len(solution)))
            solved_2 = solved_1 or all(solution[i]['attempt_2'] == gt[i] for i in range(len(solution)))

            if solved_1:
                n_correct_1 += 1
            if solved_2:
                n_correct_2 += 1
            n_total += 1
        except:
            continue

    net.hypernetwork.requires_grad_(True)
    if net.embedding_predictor:
        net.embedding_predictor.requires_grad_(True)

    pass_1 = 100.0 * n_correct_1 / n_total if n_total > 0 else 0.0
    pass_2 = 100.0 * n_correct_2 / n_total if n_total > 0 else 0.0

    print(f"\nValidation: Pass@1={pass_1:.2f}% ({n_correct_1}/{n_total}), Pass@2={pass_2:.2f}% ({n_correct_2}/{n_total})")

    return {'pass@1': pass_1, 'pass@2': pass_2, 'n_correct_1': n_correct_1, 'n_correct_2': n_correct_2, 'n_total': n_total}


if __name__ == '__main__':
    emb_dim = 128
    learning_rate = 0.02
    min_learning_rate = 0.01
    n_iterations_per_task = 20
    output_dir = './hypernetwork_outputs_augmented'
    validation_interval = 10
    validation_steps = 800
    validation_max_tasks = 20

    # Augmentation settings
    use_augmentation = True
    augmentation_types = ['original', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']

    os.makedirs(output_dir, exist_ok=True)

    if use_augmentation:
        task_names = load_tasks_with_augmentation('training', augmentation_types)
        print(f"Training with augmentation: {len(augmentation_types)} variants per task")
        print(f"Total augmented tasks: {len(task_names)}")
    else:
        task_names = load_tasks('training')
        print(f"Training on {len(task_names)} tasks (no augmentation)")

    print("Training will continue until Ctrl+C. LR will reach 0.01 after ~50 epochs and stay there.")

    net = ARCPrimaryNetwork(emb_dim=emb_dim).cuda()
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, betas=(0.5, 0.9), weight_decay=1e-3)

    iterations_per_epoch = n_iterations_per_task * len(task_names)
    gamma = 0.8409
    milestones = [
        int(iterations_per_epoch * 7.5),
        int(iterations_per_epoch * 15),
        int(iterations_per_epoch * 22.5),
        int(iterations_per_epoch * 30),
    ]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    checkpoint_path = os.path.join(output_dir, 'hypernetworks_arc.pth')
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")

    best_accuracy = 0.0
    validation_history = []
    if os.path.exists(os.path.join(output_dir, 'validation_log.json')):
        with open(os.path.join(output_dir, 'validation_log.json'), 'r') as f:
            validation_history = json.load(f)
            if validation_history:
                best_accuracy = max(v['pass@2'] for v in validation_history)

    epoch = start_epoch
    try:
        while True:
            epoch += 1
            print(f"\nEpoch {epoch} (LR: {optimizer.param_groups[0]['lr']:.4f})")

            random.seed(42 + epoch)
            epoch_tasks = task_names.copy()
            random.shuffle(epoch_tasks)

            for task_idx, task_name in enumerate(epoch_tasks):
                if (task_idx + 1) % 100 == 0:
                    print(f"  Task {task_idx+1}/{len(task_names)}")

                try:
                    if use_augmentation:
                        train_single_task_augmented(task_name, 'training', n_iterations_per_task, emb_dim,
                                                   learning_rate, net, optimizer, lr_scheduler,
                                                   save_solution=False, output_dir=output_dir)
                    else:
                        train_single_task(task_name, 'training', n_iterations_per_task, emb_dim, learning_rate,
                                        net, optimizer, lr_scheduler, save_solution=False)
                except Exception as e:
                    print(f"  ERROR: {task_name}: {e}")
                    continue

                if optimizer.param_groups[0]['lr'] < min_learning_rate:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = min_learning_rate

            torch.save({'net': net.state_dict(), 'emb_dim': emb_dim, 'epoch': epoch}, checkpoint_path)
            print(f"Saved checkpoint: epoch {epoch}")

            if validation_interval > 0 and epoch % validation_interval == 0:
                print(f"\nValidating at epoch {epoch}...")
                metrics = validate(net, 'evaluation', validation_steps, validation_max_tasks, emb_dim, 0.01)

                validation_history.append({'epoch': epoch, **metrics})

                with open(os.path.join(output_dir, 'validation_log.json'), 'w') as f:
                    json.dump(validation_history, f, indent=2)

                if metrics['pass@2'] > best_accuracy:
                    best_accuracy = metrics['pass@2']
                    torch.save({'net': net.state_dict(), 'emb_dim': emb_dim, 'epoch': epoch,
                               'val_pass@1': metrics['pass@1'], 'val_pass@2': metrics['pass@2']},
                              os.path.join(output_dir, 'best_model.pth'))
                    print(f"New best model: {best_accuracy:.2f}%")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving final checkpoint...")
        torch.save({'net': net.state_dict(), 'emb_dim': emb_dim, 'epoch': epoch}, checkpoint_path)
        print("Training stopped.")
