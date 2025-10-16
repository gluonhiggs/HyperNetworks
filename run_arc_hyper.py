import os
import json
import random
import torch
import torch.optim as optim

from train_arc_hyper import train_single_task
from primary_net_arc import ARCPrimaryNetwork


def load_tasks(split):
    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    return list(problems.keys())


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
    output_dir = './hypernetwork_outputs'
    validation_interval = 10
    validation_steps = 800
    validation_max_tasks = 20

    os.makedirs(output_dir, exist_ok=True)

    task_names = load_tasks('training')
    print(f"Training on {len(task_names)} tasks")
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
