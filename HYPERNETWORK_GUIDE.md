# HyperNetworks for CompressARC

This guide explains how to use the hypernetwork implementation for ARC-AGI puzzle solving with CompressARC.

## Overview

This implementation applies the HyperNetworks concept (Ha et al., ICLR 2017) to the CompressARC architecture. Instead of randomly initializing weights for each task, we use a **hypernetwork** that generates task-specific weights from learned puzzle embeddings.

### Key Concept

```
Task Metadata + Puzzle Embedding → HyperNetwork → All Model Weights → CompressARC → Solution
```

### Architecture Components

1. **HyperNetworkARC** (`hypernetwork_arc.py`)
   - Neural network that outputs weights for another neural network
   - Input: Puzzle embedding (128-dim vector) + normalized metadata
   - Shared body: 2-layer MLP [132 → 512 → 512]
   - Modular heads: Separate generators for each weight component type
   - Output: Complete weights_list matching ARCCompressor structure

2. **ARCCompressorHyper** (`arc_compressor_hyper.py`)
   - Modified ARCCompressor that uses hypernetwork-generated weights
   - Forward pass identical to original CompressARC
   - Enables weight sharing across tasks via shared hypernetwork

3. **Embedding Manager** (`embedding_manager.py`)
   - Manages task-specific puzzle embeddings
   - Handles save/load operations
   - Tracks embeddings across multiple tasks

4. **Training Scripts**
   - `train_arc_hyper.py`: Single-task training
   - `run_arc_hyper.py`: Multi-task meta-learning

## Benefits

- **Weight Sharing**: Learn common structure across tasks
- **Faster Adaptation**: Pre-trained hypernetwork adapts quickly to new tasks
- **Interpretable**: Puzzle embeddings capture task-specific features
- **Scalable**: Train on many tasks to improve hypernetwork

## Installation

Ensure you have the base CompressARC environment:

```bash
pip install -r requirements.txt  # torch, torchvision, PyYAML
```

Set your dataset directory:

```bash
export ARC_DATA_DIR=/path/to/arc-agi-dataset
```

## Usage

### 1. Train Single Task with Hypernetwork

Train one task, optimizing both embedding and hypernetwork:

```bash
python train_arc_hyper.py \
    --task 007bbfb7 \
    --split training \
    --iterations 2000 \
    --emb_dim 128 \
    --lr_emb 0.02 \
    --lr_hyper 0.001 \
    --output_dir ./hypernetwork_outputs
```

**Parameters:**
- `--task`: Task name to train
- `--split`: Dataset split (training/evaluation/test)
- `--iterations`: Number of training iterations
- `--emb_dim`: Puzzle embedding dimension (64-256 recommended)
- `--lr_emb`: Learning rate for puzzle embedding
- `--lr_hyper`: Learning rate for hypernetwork (0 = freeze)
- `--hypernetwork_path`: Path to pre-trained hypernetwork (optional)
- `--output_dir`: Output directory

**Output:**
- `hypernetwork_<task>.pth`: Trained hypernetwork weights
- `embedding_<task>.pth`: Trained puzzle embedding
- `solution_<task>.json`: Predicted solution

### 2. Meta-Train Across Multiple Tasks

Train a shared hypernetwork on multiple tasks:

```bash
python run_arc_hyper.py \
    --mode meta_train \
    --split training \
    --max_tasks 50 \
    --iterations 2000 \
    --emb_dim 128 \
    --lr_emb 0.02 \
    --lr_hyper 0.001 \
    --output_dir ./hypernetwork_outputs \
    --save_interval 10
```

**Parameters:**
- `--mode`: Training mode (meta_train/inference)
- `--max_tasks`: Maximum number of tasks to train (None = all)
- `--save_interval`: Save checkpoint every N tasks

**Output:**
- `hypernetwork_shared.pth`: Shared hypernetwork trained on all tasks
- `embeddings/`: Directory with all task embeddings
- `embeddings/metadata.json`: Metadata about embeddings
- `all_solutions.json`: Solutions for all tasks

**Training Strategy:**
The meta-training process:
1. Initialize random hypernetwork
2. For each task:
   - Initialize task-specific embedding
   - Optimize embedding + hypernetwork on this task
   - Save embedding and updated hypernetwork
3. Hypernetwork learns to generate good weights from embeddings

### 3. Inference with Pre-trained Hypernetwork

Use a pre-trained hypernetwork to solve new tasks:

```bash
python run_arc_hyper.py \
    --mode inference \
    --task 007bbfb7 \
    --split evaluation \
    --hypernetwork_path ./hypernetwork_outputs/hypernetwork_shared.pth \
    --iterations 2000 \
    --output_dir ./hypernetwork_outputs
```

**Inference modes:**
- **With frozen hypernetwork**: Optimize only the puzzle embedding (faster)
- **With fine-tuning**: Optimize embedding + hypernetwork (better performance)

Set `--lr_hyper 0` to freeze hypernetwork during inference.

## Workflow Examples

### Example 1: Train on Small Dataset

Train hypernetwork on first 10 tasks, then test on new task:

```bash
# Meta-train on 10 tasks
python run_arc_hyper.py --mode meta_train --max_tasks 10 --split training

# Test on new task with frozen hypernetwork
python run_arc_hyper.py \
    --mode inference \
    --task <new_task_id> \
    --split evaluation \
    --hypernetwork_path ./hypernetwork_outputs/hypernetwork_shared.pth
```

### Example 2: Continue Training

Resume meta-training from checkpoint:

```bash
# Will automatically load hypernetwork_shared.pth if it exists
python run_arc_hyper.py --mode meta_train --max_tasks 100 --split training
```

### Example 3: Single Task Exploration

Train a single task with different hyperparameters:

```bash
# Try different embedding dimensions
for EMB_DIM in 64 128 256; do
    python train_arc_hyper.py \
        --task 007bbfb7 \
        --emb_dim $EMB_DIM \
        --output_dir ./outputs_emb${EMB_DIM}
done
```

## Architecture Details

### HyperNetwork Structure

```
Input: [puzzle_emb(128) + metadata(4)] = 132-dim
   ↓
Shared Body:
   Linear(132 → 512) + ReLU
   Linear(512 → 512) + ReLU
   ↓
Modular Heads:
   - head_posterior(512 → 16384)    # For multiposteriors
   - head_linear(512 → 16384)       # For multilinear weights
   - head_direction_share(512 → 16384)  # For directional communication
   ↓
Weight Generation:
   - Reshape chunks into specific weight shapes
   - Apply Xavier-like scaling
   - Create MultiTensor structures
   ↓
Post-Processing:
   - Apply symmetrization for equivariance
   - Set requires_grad=True
   ↓
Output: Complete weights_list for ARCCompressor
```

### Metadata Normalization

Task metadata is normalized to [0, 1] range:
- `n_examples` / 12.0 (max 12 in dataset)
- `n_colors` / 20.0 (safe maximum)
- `n_x` / 30.0 (max grid height)
- `n_y` / 30.0 (max grid width)

### Dynamic Weight Generation

The hypernetwork generates different numbers of weights for different tasks:
- Iterates over all valid dims in multitensor_system
- Computes exact shapes using `multitensor_system.shape(dims, channel_dim)`
- Generates appropriately sized tensors for each dims

### Symmetrization

Post-generation, weights are symmetrized for equivariance:
- **XY Symmetry**: Swapping x and y dimensions produces swapped output
- **Directional Symmetry**: Rotations/reflections handled via weight tying

## Hyperparameters

### Recommended Settings

| Parameter | Single Task | Meta-Training | Inference |
|-----------|-------------|---------------|-----------|
| `emb_dim` | 128 | 128 | 128 |
| `lr_emb` | 0.02 | 0.02 | 0.02 |
| `lr_hyper` | 0.001 | 0.001 | 0.0 (frozen) |
| `iterations` | 2000 | 2000 | 1000-2000 |
| `hidden_dim` | 512 | 512 | 512 |

### Tuning Guidelines

- **Embedding dimension**:
  - Too small (< 64): May not capture task complexity
  - Too large (> 256): Slower, may overfit
  - Sweet spot: 128-256

- **Learning rates**:
  - `lr_emb` controls how fast embeddings adapt (0.01-0.05)
  - `lr_hyper` controls hypernetwork updates (0.0001-0.01)
  - Lower `lr_hyper` for stable meta-learning

- **Iterations**:
  - Single task: 2000 sufficient (same as original CompressARC)
  - With pre-trained hypernetwork: 1000 may suffice

## Comparison with Original CompressARC

| Aspect | Original CompressARC | HyperNetwork CompressARC |
|--------|----------------------|--------------------------|
| Initialization | Random per task | Generated from embedding |
| Training | Optimize weights directly | Optimize embedding + hypernetwork |
| Memory | ~27 weight components | Same + hypernetwork (~1M params) |
| Speed | Fast (single task) | Slower (generates weights each forward) |
| Generalization | No transfer | Learns across tasks |
| Interpretability | Low | High (via embeddings) |

## File Structure

```
hypernetwork_arc.py          # Core hypernetwork module
arc_compressor_hyper.py      # Modified ARCCompressor
embedding_manager.py         # Embedding management utilities
train_arc_hyper.py          # Single-task training script
run_arc_hyper.py            # Multi-task meta-learning script
HYPERNETWORK_GUIDE.md       # This file
```

## Troubleshooting

### Out of Memory

- Reduce `emb_dim` (e.g., 64 instead of 128)
- Reduce batch size (not applicable for single-task training)
- Use gradient checkpointing (requires code modification)

### Poor Performance

- Increase `iterations` (try 3000-5000)
- Adjust learning rates (try lr_emb=0.05, lr_hyper=0.0001)
- Train hypernetwork on more tasks before inference
- Check that metadata is normalized correctly

### Slow Training

- The hypernetwork generates weights every forward pass, which is slower than original CompressARC
- For production, consider caching generated weights when embedding doesn't change
- Use smaller `hidden_dim` (256 instead of 512)

## Future Improvements

1. **Fast Inference Mode**: Cache generated weights when embedding is fixed
2. **Adaptive Architecture**: Generate different architectures for different tasks
3. **Hierarchical Embeddings**: Capture task families via embedding clustering
4. **Meta-Learning Algorithms**: Implement MAML, Reptile for better generalization
5. **Multi-GPU Meta-Training**: Parallelize across tasks and GPUs

## References

- Ha et al., "HyperNetworks", ICLR 2017 ([arxiv](https://arxiv.org/abs/1609.09106))
- Original CompressARC: `COMPRESSARC.md`
- Original HyperNetworks implementation: `train_hyper.py`, `hypernetwork_modules.py`

## Citation

If you use this code, please cite both the HyperNetworks paper and the CompressARC work.

## Contact

For issues or questions, please refer to the main repository README or open an issue on GitHub.
