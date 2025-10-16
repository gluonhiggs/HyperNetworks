# Quick Start Guide - Dynamic HyperNetwork for CompressARC

Get up and running with the Dynamic HyperNetwork in 30 minutes.

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

## Installation (5 minutes)

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install other dependencies
pip install numpy tqdm

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Dataset Setup (10 minutes)

Download the ARC dataset:
```bash
# Create dataset directory
mkdir -p dataset

# Download ARC-AGI dataset
# Place arc-agi_training_challenges.json and arc-agi_training_solutions.json in dataset/
```

## Verify Installation (2 minutes)

Run the integration tests:
```bash
python tests/test_integration.py
```

Expected output:
```
✓ Test 1/6: Import validation... PASSED
✓ Test 2/6: Task preprocessing... PASSED
✓ Test 3/6: Model initialization... PASSED
✓ Test 4/6: Forward pass... PASSED
✓ Test 5/6: Training step... PASSED
✓ Test 6/6: Solution generation... PASSED

All tests passed! (6/6)
```

## First Training Run (10 minutes)

Train on a single task:
```bash
python src/train_hypernetwork.py \
    --data_dir dataset/ \
    --task_names 00d62c1b \
    --num_epochs 50 \
    --device cuda
```

Expected output:
```
Loading tasks from dataset/...
Loaded 1 tasks
Creating hypernetwork...
Model has 1,523,456 parameters
Starting training on 1 tasks...
Epoch 1/50 - Avg Loss: 0.0524 - LR: 0.001000
...
Training complete!
```

## Quick Usage Example

```python
from src.arc_hypercompressor import HyperCompressorFactory

# Create model
model = HyperCompressorFactory.create_default(device='cuda')

# Generate task-specific weights
task_name = "00d62c1b"
task_metadata = {
    'n_examples': 4,
    'n_colors': 3,
    'n_x': 10,
    'n_y': 10,
    'n_train': 3,
    'n_test': 1,
}

# Get embedding and weights
embedding, weights = model(task_name, task_metadata)
print(f"Embedding shape: {embedding.shape}")  # [1, 64]
print(f"Generated {len(weights)} weight components")
```

## Next Steps

- Read the [full training guide](training_guide.md)
- Explore [example usage](../src/example_usage.py)
- See [architecture documentation](hypernetwork_architecture.md)

## Common Issues

**CUDA out of memory:**
```bash
# Use smaller batch size or CPU
python src/train_hypernetwork.py --batch_size 1 --device cpu
```

**Import errors:**
```bash
# Make sure you're in the project root
cd /path/to/HyperNetworks
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Dataset not found:**
```bash
# Set data directory explicitly
export ARC_DATA_DIR=/path/to/dataset
python src/train_hypernetwork.py --data_dir $ARC_DATA_DIR
```
