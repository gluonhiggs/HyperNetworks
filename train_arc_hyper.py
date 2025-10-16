import os
import json
import torch
import torch.optim as optim
import numpy as np

import pre_processing
import train
from primary_net_arc import ARCPrimaryNetwork
import solution_selection


def train_single_task(task_name, split, n_iterations, emb_dim, lr,
                      net, optimizer, lr_scheduler=None, save_solution=False, output_dir='./hypernetwork_outputs'):
    """Train one ARC task with hypernetwork."""

    torch.set_default_device('cuda')
    device = torch.device('cuda')

    seed = hash(task_name) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_dir = os.environ.get('ARC_DATA_DIR', 'dataset/')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    task = pre_processing.Task(task_name, problems[task_name], None)
    del problems

    _ = net.get_or_create_embedding(task_name, task)

    if save_solution:
        train_history_logger = solution_selection.Logger(task)
    else:
        class LightweightLogger:
            def __init__(self):
                self.loss_curve = []
            def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss):
                self.loss_curve.append(float(loss.detach().cpu().numpy()))
        train_history_logger = LightweightLogger()

    class ModelWrapper:
        def __init__(self, net, task, task_name):
            self.net = net
            self.task = task
            self.task_name = task_name
        def forward(self):
            return self.net(self.task, self.task_name)

    model = ModelWrapper(net, task, task_name)

    for train_step in range(n_iterations):
        train.take_step(task, model, optimizer, train_step, train_history_logger, n_iterations, lr_scheduler, None)

    if save_solution and hasattr(train_history_logger, 'solution_most_frequent'):
        example_list = []
        for example_num in range(task.n_test):
            attempt_1 = [list(row) for row in train_history_logger.solution_most_frequent[example_num]]
            attempt_2 = [list(row) for row in train_history_logger.solution_second_most_frequent[example_num]]
            example_list.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})

        os.makedirs(output_dir, exist_ok=True)
        solution_path = os.path.join(output_dir, f'solution_{task_name}.json')
        with open(solution_path, 'w') as f:
            json.dump({task_name: example_list}, f, indent=2)

        return example_list

    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train single ARC task with hypernetwork')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--split', type=str, default='training')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default='./hypernetwork_outputs')

    args = parser.parse_args()

    net = ARCPrimaryNetwork(emb_dim=args.emb_dim).cuda()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.0005)

    train_single_task(
        task_name=args.task,
        split=args.split,
        n_iterations=args.iterations,
        emb_dim=args.emb_dim,
        lr=args.lr,
        net=net,
        optimizer=optimizer,
        lr_scheduler=None,
        save_solution=True,
        output_dir=args.output_dir
    )
