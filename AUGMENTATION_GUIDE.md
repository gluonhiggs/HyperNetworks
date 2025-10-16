# Data Augmentation for Hypernetwork Training

## Overview

This guide explains how to use data augmentation to increase training diversity for ARC-AGI hypernetwork meta-learning.

## Why Augmentation Creates Genuinely Different Tasks

In ARC-AGI, each task defines a **transformation rule** (input→output mapping). When you augment a task through rotation or flipping:

- **Original task**: "Add red pixel to the RIGHT of blue pixel"
- **After 90° rotation**: "Add red pixel BELOW blue pixel"
- **After flip**: "Add red pixel to the LEFT of blue pixel"

These are **different transformation rules** in absolute coordinate space. Unlike image classification (where a rotated cat is still a cat), **rotation changes the rule itself**.

## Augmentation Types

The augmented training uses 6 variants per task:

1. `original` - No transformation
2. `rot90` - Rotate 90° clockwise
3. `rot180` - Rotate 180°
4. `rot270` - Rotate 270° clockwise (90° counter-clockwise)
5. `flip_h` - Horizontal flip
6. `flip_v` - Vertical flip

**Result**: 1,000 original tasks → 6,000 genuinely diverse transformation rules

## Files

### `run_arc_hyper_augmented.py`

Augmented version of the meta-training script with the following additions:

1. **`load_tasks_with_augmentation()`** - Creates augmented task list
2. **`parse_augmented_task_name()`** - Parses names like `c9680e90_rot90`
3. **`apply_augmentation()`** - Applies transformation to task dictionary
4. **`train_single_task_augmented()`** - Training wrapper with augmentation support

### Configuration

```python
# In run_arc_hyper_augmented.py (lines 120-122)
use_augmentation = True
augmentation_types = ['original', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']
```

### Output Directory

- **Non-augmented**: `./hypernetwork_outputs/`
- **Augmented**: `./hypernetwork_outputs_augmented/`

This prevents conflicts between the two training approaches.

## Usage

### Start Augmented Training

```bash
python run_arc_hyper_augmented.py
```

### Disable Augmentation

Set `use_augmentation = False` in line 121.

## How It Works

1. **Task naming**: Each augmented variant gets unique name:
   - Original: `c9680e90`
   - Rotated 90°: `c9680e90_rot90`
   - Flipped horizontal: `c9680e90_flip_h`

2. **Separate embeddings**: Each variant gets its own embedding in the hypernetwork, since they represent different transformation rules.

3. **Deterministic augmentation**: Fixed seed mapping ensures:
   - `rot90` always uses seed=1
   - `rot180` always uses seed=2
   - etc.

4. **Meta-learning**: Hypernetwork learns to generate weights for 6× more diverse transformation rules, improving generalization.

## Training Time Estimate

- **Without augmentation**: 1,000 tasks × 20 iterations = 20,000 total iterations
- **With augmentation**: 6,000 tasks × 20 iterations = 120,000 total iterations
- **Time multiplier**: ~6× longer per epoch

With 20 iterations/task at ~2.5 min/task:
- 6,000 tasks × 2.5 min = 15,000 min ≈ 250 hours ≈ 10.4 days per epoch

**Recommendation**: Run for multiple weeks if possible, or reduce `augmentation_types` to fewer variants.

## Benefits

1. ✓ **6× more transformation rules** to learn from
2. ✓ **Better generalization** to arbitrary orientations at test time
3. ✓ **Genuine diversity** - not superficial data augmentation
4. ✓ **Proper meta-learning** - hypernetwork sees more rule patterns

## Comparison

| Metric | Non-augmented | Augmented |
|--------|---------------|-----------|
| Tasks per epoch | 1,000 | 6,000 |
| Distinct rules | 1,000 | 6,000 |
| Time per epoch | ~42 hours | ~250 hours |
| Generalization | Good | Better |

## Notes

- Current training (`run_arc_hyper.py`) is **unaffected** by file edits
- Augmented version saves to separate directory
- Validation still uses non-augmented evaluation tasks (for fair comparison)
