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

2. **ARCPrimaryNetwork** (`arc_primary_net.py`)
   - End-to-end model containing hypernetwork, embeddings, AND architecture (matching CIFAR-10 PrimaryNetwork pattern)
   - Manages task-specific puzzle embeddings via ModuleDict
   - Single state_dict for save/load operations (hypernetwork + embeddings together)
   - Implements complete forward pass from task → weights → output

3. **Training Scripts**
   - `train_arc_hyper.py`: Single-task training or inference with test-time adaptation
   - `run_arc_hyper.py`: Multi-task meta-training across all training tasks

## Benefits

- **Meta-Learning**: Shared hypernetwork learns cross-task patterns (embeddings are task-specific, NOT shared)
- **Faster Adaptation**: Pre-trained hypernetwork provides better initialization for new tasks
- **Fewer Parameters**: Optimize only embedding (~128 params) instead of all weights (~100K+)
- **Better Convergence**: Test-time training converges faster due to learned inductive biases
- **Interpretable**: Puzzle embeddings capture task-specific features

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
    --lr 0.01 \
    --output_dir ./hypernetwork_outputs
```

**Parameters:**
- `--task`: Task name to train
- `--split`: Dataset split (training/evaluation/test)
- `--iterations`: Number of training iterations
- `--emb_dim`: Puzzle embedding dimension (64-256 recommended)
- `--lr`: Learning rate (single LR for both embedding and hypernetwork, matching CIFAR-10)
- `--freeze_hypernetwork`: Freeze hypernetwork (for test-time inference)
- `--hypernetwork_path`: Path to pre-trained hypernetwork (optional)
- `--output_dir`: Output directory

**Output:**
- `hypernetwork_<task>.pth`: Trained hypernetwork weights
- `embedding_<task>.pth`: Trained puzzle embedding
- `solution_<task>.json`: Predicted solution

### 2. Meta-Train Across Multiple Tasks

Train a shared hypernetwork on multiple tasks using multi-epoch training:

```bash
python run_arc_hyper.py           # Train on all training tasks
python run_arc_hyper.py --resume  # Resume from checkpoint
```

**Configuration** (hardcoded in script, matching CIFAR-10 pattern):
- `emb_dim`: 128
- `lr`: 0.01 (single LR for both embedding and hypernetwork)
- `n_epochs`: 50
- `n_iterations_per_task`: 100 (per epoch)
- `save_interval`: 1 (save every epoch)

**Output:**
- `hypernetworks_arc.pth`: ARCPrimaryNetwork checkpoint (hypernetwork + all embeddings)
- `all_solutions.json`: Solutions for all tasks

**Training Strategy (Multi-Epoch):**
The meta-training process avoids catastrophic forgetting:
1. Initialize ARCPrimaryNetwork (hypernetwork + empty task embeddings dict)
2. For each epoch (50 total):
   - For each task:
     - Get/create task-specific embedding
     - Short optimization (100 iters) on this task
     - Update both embedding and hypernetwork weights
   - Save checkpoint after epoch
3. Hypernetwork learns cross-task weight generation patterns
4. Each task receives 50 × 100 = 5000 total optimization steps (2.5× baseline)

### 3. Inference with Pre-trained Hypernetwork

Use a pre-trained hypernetwork to solve new tasks:

```bash
python train_arc_hyper.py \
    --task 007bbfb7 \
    --split evaluation \
    --freeze_hypernetwork \
    --hypernetwork_path ./hypernetwork_outputs/hypernetworks_arc.pth \
    --iterations 2000 \
    --output_dir ./hypernetwork_outputs
```

**Inference modes:**
- **With frozen hypernetwork**: Use `--freeze_hypernetwork` flag (optimize only embedding, faster)
- **With fine-tuning**: Omit flag (optimize embedding + hypernetwork, better performance)

## Workflow Examples

### Example 1: Standard Training Workflow

Meta-train on all training tasks, then evaluate:

```bash
# Meta-train on all training tasks (50 epochs)
python run_arc_hyper.py

# Test on new evaluation task with frozen hypernetwork
python train_arc_hyper.py \
    --task <eval_task_id> \
    --split evaluation \
    --freeze_hypernetwork \
    --hypernetwork_path ./hypernetwork_outputs/hypernetworks_arc.pth
```

### Example 2: Resume Training

Resume meta-training from checkpoint:

```bash
# Will automatically load hypernetworks_arc.pth if it exists
python run_arc_hyper.py --resume
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
- `n_colors` / 9.0 (safe maximum)
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
| `lr` | 0.01 | 0.01 | 0.01 |
| `freeze_hypernetwork` | False | False | True |
| `iterations` | 2000 | 100 (per task per epoch) | 1000-2000 |
| `n_epochs` | N/A | 50 | N/A |
| `hidden_dim` | 512 | 512 | 512 |

### Tuning Guidelines

- **Embedding dimension**:
  - Too small (< 64): May not capture task complexity
  - Too large (> 256): Slower, may overfit
  - Sweet spot: 128-256

- **Learning rate**:
  - Single unified LR for both embedding and hypernetwork (matching CIFAR-10 pattern)
  - Original CompressARC uses lr=0.01 (baseline)
  - Range: 0.005-0.02

- **Iterations**:
  - Single task: 2000 sufficient (same as original CompressARC)
  - Meta-training: 100 per task per epoch × 50 epochs = 5000 total per task
  - With pre-trained hypernetwork: 1000 may suffice

## Comparison with Original CompressARC

| Aspect | Original CompressARC | HyperNetwork (Meta-Training) | HyperNetwork (Inference) |
|--------|----------------------|------------------------------|--------------------------|
| Initialization | Random weights | Random weights | Structured from trained hypernetwork |
| Parameters optimized | All weights (~100K+) | Embedding + hypernetwork | Embedding only (~128) |
| Iterations needed | 2000 per task | 100 per task per epoch | 1000-1500 per task |
| Cross-task learning | None | Yes (shared hypernetwork) | Benefits from meta-training |
| Memory | ~27 weight components | Same + hypernetwork (~1M) | Same + hypernetwork (~1M) |
| Generalization | No transfer | Learns across tasks | Better initialization |
| Test-time training | Required | N/A (training phase) | Required (but faster) |

## File Structure

```
hypernetwork_arc.py          # Core hypernetwork module (generates weights)
arc_primary_net.py           # End-to-end model (hypernetwork + embeddings + architecture)
train_arc_hyper.py          # Single-task training / inference script
run_arc_hyper.py            # Multi-epoch meta-training script
README.md                   # This file (ARC hypernetworks guide)
README_CIFAR10.md           # Original CIFAR-10 hypernetwork implementation
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
