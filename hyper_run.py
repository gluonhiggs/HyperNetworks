import os
import sys
import time
import json
import multiprocessing
import gc

import numpy as np
import torch

import solve_task_hyper


KAGGLE = False
if KAGGLE:
    time_limit = 12
    split = "test"
    os.environ['ARC_DATA_DIR'] = '/kaggle/input/arc-prize-2025/'
else:
    splits = ["sample", "training", "sample20", "training20", "evaluation", "test"]
    split_type = 4  # 0=sample, 1=training, 2=sample20, 3=training20, 4=evaluation, 5=test
    time_limits = [1, 64, 3, 3, 18, 18]
    split = splits[split_type]
    time_limit = time_limits[split_type]
    os.environ['ARC_DATA_DIR'] = 'dataset/'


multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True


def parallelize_runs(gpu_quotas, task_usages, n_iterations, checkpoint_path, emb_dim, verbose=False):
    """
    Run trained hypernetwork on tasks in parallel across GPUs.

    Args:
        gpu_quotas: Available memory per GPU (bytes)
        task_usages: Estimated memory per task (bytes)
        n_iterations: Training steps per task
        checkpoint_path: Path to trained hypernetwork checkpoint
        emb_dim: Embedding dimension
        verbose: Print scheduling details
    """
    gpu_quotas = gpu_quotas[:]
    n_gpus = len(gpu_quotas)
    n_tasks = len(task_names)

    tasks_started = [False] * n_tasks
    tasks_finished = [False] * n_tasks
    processes = [None] * n_tasks
    process_gpu_ids = [None] * n_tasks
    gpu_task_counts = [0] * n_gpus

    t = time.time()

    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()

        while not all(tasks_finished):
            if not error_queue.empty():
                raise ValueError(error_queue.get())

            # Check finished tasks
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(f"{task_names[i]} finished on GPU {process_gpu_ids[i]} "
                                  f"(quota: {gpu_quotas[process_gpu_ids[i]]/1024**3:.1f} GB)")

            # Schedule tasks (deterministic memory-descending order)
            global_task_order = sorted(range(n_tasks), key=lambda i: task_usages[i], reverse=True)
            gpu_order = sorted(range(n_gpus),
                             key=lambda g: (gpu_quotas[g], -gpu_task_counts[g]),
                             reverse=True)

            for gpu_id in gpu_order:
                for i in global_task_order:
                    enough_quota = gpu_quotas[gpu_id] > task_usages[i]
                    running_tasks = sum(tasks_started) - sum(tasks_finished)
                    enough_cpus = running_tasks < n_cpus

                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        gpu_task_counts[gpu_id] += 1

                        args = (task_names[i], split, end_time, n_iterations, gpu_id,
                               memory_dict, solutions_dict, error_queue, checkpoint_path, emb_dim)
                        p = multiprocessing.Process(target=solve_task_hyper.solve_task_hyper, args=args)
                        p.start()

                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id

                        if verbose:
                            print(f"{task_names[i]} started on GPU {gpu_id} "
                                  f"(quota: {gpu_quotas[gpu_id]/1024**3:.1f} GB)")

            time.sleep(0.5)

        if not error_queue.empty():
            raise ValueError(error_queue.get())

        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)

    time_taken = time.time() - t
    if verbose:
        print(f'\nAll tasks finished in {time_taken:.1f} seconds.')

    return memory_dict, solutions_dict, time_taken


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    start_time = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Running trained hypernetwork on split: {split}")

    end_time = start_time + time_limit * 3600 - 120

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Load task names
    data_dir = os.environ.get('ARC_DATA_DIR')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    task_names = list(problems.keys())
    del problems
    n_tasks = len(task_names)

    print(f"Tasks: {n_tasks}, GPUs: {n_gpus}, CPUs: {n_cpus}")

    # Hypernetwork configuration
    checkpoint_path = './hypernetwork_outputs/hypernetworks_arc.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = './hypernetwork_outputs/best_model.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found in ./hypernetwork_outputs/")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    emb_dim = checkpoint.get('emb_dim', 128)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Embedding dim: {emb_dim}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'val_pass@2' in checkpoint:
        print(f"Validation Pass@2: {checkpoint['val_pass@2']:.2f}%")
    del checkpoint

    # GPU memory setup
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    safe_gpu_memory_quotas = [int(quota - 8 * 1024**3) for quota in gpu_memory_quotas]

    # Memory profiling run
    print("\nRunning memory profiling...")
    initial_task_usages = [2 * 1024**3] * n_tasks
    memory_dict, _, _ = parallelize_runs(
        safe_gpu_memory_quotas, initial_task_usages, 2,
        checkpoint_path, emb_dim, verbose=True
    )

    # Sort tasks by memory (deterministic scheduling)
    memory_quantum = 256 * 1024**2  # 256MB buckets
    tasks = sorted(memory_dict.items(),
                   key=lambda x: (-(x[1] // memory_quantum), x[0]),
                   reverse=False)
    task_names, task_memory_usages = zip(*tasks)

    torch.cuda.empty_cache()
    gc.collect()

    # Main inference run
    print("\nRunning main inference...")
    n_steps = 800  # Test-time adaptation steps (fewer than training)
    print(f"Inference steps per task: {n_steps}")

    main_task_usages = [int(u * 1.2) for u in task_memory_usages]
    _, solutions_dict, time_taken = parallelize_runs(
        safe_gpu_memory_quotas, main_task_usages, n_steps,
        checkpoint_path, emb_dim, verbose=True
    )

    # Save solutions
    output_file = f'submission_hyper_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(solutions_dict, f, separators=(', ', ': '))

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed: {n_tasks} tasks")
    print(f"Steps per task: {n_steps}")
    print(f"Inference time: {time_taken:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
