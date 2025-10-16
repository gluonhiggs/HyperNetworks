# %%
import os
import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool
import gc

import numpy as np
import torch

# %%
KAGGLE = False
if KAGGLE:
    time_limit = 12
    split = "test"
    os.environ['ARC_DATA_DIR'] = '/kaggle/input/arc-prize-2025/'
    sys.path.append('/kaggle/input/compressarc-upgraded-1')
else:
    # change to 0.3 if run sample
    splits = ["sample", "training", "sample20", "training20", "evaluation"]
    split_type = 2 # 0 for sample, 1 for training, 2 for sample20, 3 for training20, 4 for evaluation
    time_limits = [1, 64, 3, 3, 18]
    split = splits[split_type]
    time_limit = time_limits[split_type]
    # split = "training"
    os.environ['ARC_DATA_DIR'] = 'dataset_old'

# %%
import solve_task
# import solve_task_maml as solve_task






# %%
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
# Seeds moved inside if __name__ for deterministic main process initialization
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


    start_time = time.time()
    # Format time to YYYY-MM-DD HH:MM:SS then print
    format_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f'Start training at {format_start_time}')
    end_time = start_time + time_limit*3600 - 120

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Find all the puzzle names
    
    data_dir = os.environ.get('ARC_DATA_DIR')
    with open(f'{data_dir}/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    task_names = list(problems.keys())
    del problems
    n_tasks = len(task_names)

# %%
def parallelize_runs(gpu_quotas, task_usages, n_iterations, verbose=False):
    gpu_quotas = gpu_quotas[:]
    # Schedule the tasks greedily to max out memory usage
    t = time.time()
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]
    gpu_task_counts = [0 for _ in range(n_gpus)]  # Track tasks per GPU for load balancing
    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()
        while not all(tasks_finished):
            if not error_queue.empty():
                raise ValueError(error_queue.get())
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(task_names[i], 'finished on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])
            # Sort by memory descending for consistent scheduling
            global_task_order = sorted(range(n_tasks), key=lambda i: task_usages[i], reverse=True)
            # REPRODUCIBILITY FIX: Use fixed GPU order with deterministic tie-breaking
            # Primary: highest quota, Secondary: fewest tasks assigned (for load balancing)
            gpu_order = list(range(n_gpus))  # Always [0, 1, 2, ...] for determinism
            sorted_gpu_ids = sorted(gpu_order,
                                   key=lambda g: (gpu_quotas[g], -gpu_task_counts[g]),
                                   reverse=True)

            for gpu_id in sorted_gpu_ids:
                # REPRODUCIBILITY FIX: Use fixed task order (no reversal based on loop_count)
                task_order = global_task_order[:]  # Always memory-descending order
                for i in task_order:
                    enough_quota = gpu_quotas[gpu_id] > task_usages[i]
                    enough_cpus = sum(int(started) for started in tasks_started) - sum(int(finished) for finished in tasks_finished) < n_cpus
                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        gpu_task_counts[gpu_id] += 1  # Track task assignment for load balancing
                        args = (task_names[i], split, end_time, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        if verbose:
                            print(task_names[i], 'started on gpu', gpu_id,
                                  'New quota is', gpu_quotas[gpu_id])
            time.sleep(0.5)
        if not error_queue.empty():
            raise ValueError(error_queue.get())
        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)
    time_taken = time.time() - t
    if verbose:
        print('All jobs finished in', time_taken, 'seconds.')
    return memory_dict, solutions_dict, time_taken
# %%
if __name__ == '__main__':
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    safe_gpu_memory_quotas = [int((memory_quota - 8* 1024**3)) for memory_quota in gpu_memory_quotas] # Change to 16GB safety margin on kaggle
    initial_task_usages = [2 * 1024**3 for i in range(n_tasks)]
    task_usages = [1 for i in range(n_tasks)]
    memory_dict, _, _ = parallelize_runs(safe_gpu_memory_quotas, initial_task_usages, 2, verbose=True)
   
    # Sort the tasks by decreasing memory usage
    # REPRODUCIBILITY FIX: Quantize memory to 256MB buckets for stable sorting
    # Memory measurements can vary by ±50-100MB between runs due to GPU allocator
    # Quantizing reduces sensitivity while preserving memory-based scheduling efficiency
    # Secondary sort by task name for deterministic tie-breaking within same bucket
    memory_quantum = 256 * 1024**2  # 256MB buckets
    tasks = sorted(memory_dict.items(),
                   key=lambda x: (-(x[1] // memory_quantum), x[0]),  # (-memory_bucket, name)
                   reverse=False)  # False because we negate memory for descending
    task_names, task_memory_usages = zip(*tasks)
    torch.cuda.empty_cache()
    gc.collect()
# %%
# if __name__ == '__main__':
#     test_steps = 20
#     _, _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, test_steps, verbose=False)
#     import gc
#     torch.cuda.empty_cache()
#     gc.collect()
# %%
if __name__ == '__main__':
    # time_per_step = time_taken / test_steps
    # time_left = end_time - time.time()
    # n_steps = int(time_left // time_per_step)
    n_steps = 1300
    print(f'n_steps: {n_steps}')
    prep_time = time.time() - start_time
    # FIXED: Scale usages 1.2x for main phase safety (backprop variance)
    main_task_usages = [int(u * 1.2) for u in task_memory_usages]
    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, main_task_usages, n_steps, verbose=True)
    # Format the solutions and put into submission file
    with open('submission.json', 'w') as f:
        json.dump(solutions_dict, f, separators=(', ', ': '))
    total_time = prep_time + time_taken
    print(n_tasks, 'tasks solved.')
    print(n_steps, 'steps taken.')
    print(time_taken, 'seconds taken.')
    print(total_time, 'seconds total.')