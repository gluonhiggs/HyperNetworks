# GPU Memory Usage in HyperNetworks: A Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Memory Components Breakdown](#memory-components-breakdown)
3. [Memory Usage During Training](#memory-usage-during-training)
4. [Memory Bottlenecks](#memory-bottlenecks)
5. [HyperNetwork-Specific Considerations](#hypernetwork-specific-considerations)
6. [Memory Optimization Techniques](#memory-optimization-techniques)
7. [Practical Guidelines](#practical-guidelines)

---

## Overview

During neural network training on GPUs, memory is consumed by multiple components. Understanding what occupies GPU memory is critical for:
- Maximizing batch sizes
- Preventing out-of-memory (OOM) errors
- Optimizing training efficiency
- Making hardware purchasing decisions

**Key Insight:** For HyperNetworks, while parameters are drastically reduced (~440× compression), **activations still dominate memory usage**, meaning the memory savings are less dramatic than parameter count suggests.

---

## Memory Components Breakdown

### 1. **Model Parameters (Weights)**

**What:** All learnable parameters (`nn.Parameter`) that define the model.

**Size Formula:**
```
Memory = N_params × bytes_per_param
```

**Typical sizes:**
- Float32 (FP32): 4 bytes per parameter
- Float16 (FP16): 2 bytes per parameter
- BFloat16 (BF16): 2 bytes per parameter

**Examples:**
```
Standard ResNet-18:  ~11M params × 4 bytes = 44 MB
Standard ResNet-50:  ~25M params × 4 bytes = 100 MB
HyperNetwork ResNet-18: ~25k params × 4 bytes = 100 KB (440× smaller!)
```

**HyperNetwork Components:**
- Embeddings (z vectors): One per convolution layer
- HyperNetwork parameters (W₁, b₁, W₂, b₂): Shared across all layers
- Standard layers (first conv, batch norms, final linear): Same as standard networks

---

### 2. **Gradients**

**What:** Derivatives ∂Loss/∂param for each parameter, computed during backpropagation.

**Size:** Exactly the same as parameters

```
Gradient Memory = Parameter Memory
```

**Examples:**
```
Standard ResNet-18 Gradients:  44 MB
HyperNetwork ResNet-18 Gradients: 100 KB
```

**Note:** Gradients accumulate during backward pass and are cleared after `optimizer.step()` (or explicitly with `zero_grad()`).

---

### 3. **Optimizer States**

**What:** Additional statistics maintained by optimizers for adaptive learning rates.

**Size depends on optimizer:**

| Optimizer | State Size | Components Stored |
|-----------|------------|-------------------|
| **SGD** | 0 (no momentum) | None |
| **SGD + Momentum** | 1× params | Velocity |
| **Adam** | 2× params | First moment (m), Second moment (v) |
| **AdamW** | 2× params | First moment (m), Second moment (v) |
| **RMSprop** | 1× params | Moving average of squared gradients |

**Examples (using Adam):**
```
Standard ResNet-18:  2 × 44 MB = 88 MB
HyperNetwork ResNet-18: 2 × 100 KB = 200 KB
```

**Insight:** Adam uses significantly more memory than SGD, but provides better convergence for HyperNetworks due to per-parameter adaptive learning rates.

---

### 4. **Input Batch Data**

**What:** Current batch of images/data being processed.

**Size Formula:**
```
Memory = batch_size × channels × height × width × bytes_per_value
```

**Examples:**
```
CIFAR-10 (32×32, batch=128):
  128 × 3 × 32 × 32 × 4 = 1.57 MB

ImageNet (224×224, batch=256):
  256 × 3 × 224 × 224 × 4 = 154 MB

High-res images (1024×1024, batch=16):
  16 × 3 × 1024 × 1024 × 4 = 201 MB
```

**Key Factor:** Batch size is the primary control for this component.

---

### 5. **Intermediate Activations** ⭐ **USUALLY THE LARGEST COMPONENT!**

**What:** Outputs of each layer, stored during forward pass and required for backpropagation.

**Why Needed:**
```python
# Chain rule requires intermediate values:
∂Loss/∂W_layer1 = ∂Loss/∂output × ∂output/∂W_layer1
                                    ↑
                            Needs stored activations!
```

**Size Characteristics:**
- Scales **linearly with batch size**
- Scales with **network depth** (more layers = more activations)
- Scales with **spatial dimensions** (early layers have large feature maps)

**Estimation:**
For each convolutional layer:
```
Activation Memory = batch_size × out_channels × height × width × 4 bytes
```

**Example (ResNet-18 on CIFAR-10, batch=128):**
```
Layer                    Activation Size
─────────────────────────────────────────────
Conv1: 128×16×32×32      2.1 MB
Block1: 128×16×32×32     2.1 MB (× 6 blocks = ~12 MB)
Block2: 128×32×16×16     1.05 MB (× 6 blocks = ~6 MB)
Block3: 128×64×8×8       0.52 MB (× 6 blocks = ~3 MB)
─────────────────────────────────────────────
Total Activations:       ~50-100 MB
```

**Example (ResNet-50 on ImageNet, batch=256):**
```
Early layers: 256×64×112×112 = 205 MB (single layer!)
Total activations: 1-10 GB depending on architecture details
```

**Critical Insight:** This is why increasing batch size quickly leads to OOM errors!

---

### 6. **Generated Weights (HyperNetwork-Specific)**

**What:** Dynamically generated convolutional kernels created by the HyperNetwork during each forward pass.

**Size Formula:**
```
Per kernel: out_channels × in_channels × kernel_h × kernel_w × 4 bytes
```

**Example (HyperNetwork ResNet-18):**
```
36 convolutional layers:
- Smallest: 16×16×3×3 = 2,304 params = 9 KB
- Largest: 64×64×3×3 = 36,864 params = 147 KB
Total generated weights: ~5 MB
```

**Characteristics:**
- Generated fresh each forward pass
- Temporary (not stored long-term like parameters)
- Additional overhead compared to standard networks
- Still much smaller than activations

---

### 7. **Additional Components**

#### **a) Temporary Buffers**
Operations like matrix multiplication, concatenation, and reshaping may require temporary GPU memory.

**Typical size:** 10-100 MB depending on operations

#### **b) BatchNorm Statistics**
Running mean and variance for batch normalization layers.

**Size:** Small (~0.1-1 MB), negligible

#### **c) CUDA Context Overhead**
PyTorch and CUDA runtime require baseline memory.

**Size:** ~300-800 MB (varies by GPU and PyTorch version)

#### **d) cuDNN Workspace**
cuDNN allocates workspace memory for optimized convolution algorithms.

**Size:** 10-500 MB depending on convolution sizes

#### **e) PyTorch Memory Allocator Cache**
PyTorch caches freed memory for reuse (avoiding frequent malloc/free).

**Effect:** Reported GPU memory usage may be higher than actual usage

**Control:**
```python
torch.cuda.empty_cache()  # Release cached memory (rarely needed)
```

---

## Memory Usage During Training

### Training Phase Timeline

```
┌─────────────────────────────────────────────────────────────┐
│ INITIALIZATION (One-time)                                    │
├─────────────────────────────────────────────────────────────┤
│ - Allocate parameters                                        │
│ - Allocate optimizer states                                  │
│ - Allocate CUDA context                                      │
│ - Allocate cuDNN workspace                                   │
│                                                              │
│ Memory: 100 MB - 1 GB (baseline)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ FORWARD PASS (Per Batch)                                     │
├─────────────────────────────────────────────────────────────┤
│ GPU Memory Contains:                                         │
│ ✓ Parameters (persistent)                                   │
│ ✓ Input batch (loaded from CPU/disk)                        │
│ ✓ Activations (accumulating layer by layer)                 │
│ ✓ Generated weights (HyperNetwork only)                     │
│ ✓ Temporary buffers (operations)                            │
│                                                              │
│ Memory: Baseline + Batch + Activations + Generated          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ BACKWARD PASS (PEAK MEMORY!)                                │
├─────────────────────────────────────────────────────────────┤
│ GPU Memory Contains:                                         │
│ ✓ Parameters (needed for gradient computation)              │
│ ✓ Input batch (still needed)                                │
│ ✓ ALL activations (needed for chain rule!)                  │
│ ✓ Generated weights (HyperNetwork)                          │
│ ✓ Gradients (accumulating)                                  │
│ ✓ Optimizer states (accessed)                               │
│ ✓ Temporary buffers                                         │
│                                                              │
│ Memory: MAXIMUM (everything above)                          │
│                                                              │
│ This is when OOM errors typically occur!                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ OPTIMIZER STEP                                               │
├─────────────────────────────────────────────────────────────┤
│ GPU Memory Contains:                                         │
│ ✓ Parameters (being updated)                                │
│ ✓ Gradients                                                  │
│ ✓ Optimizer states (being updated)                          │
│                                                              │
│ Activations freed! Batch freed! Generated weights freed!    │
│                                                              │
│ Memory: Back to baseline + params + grads + opt states      │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    Next batch begins
```

---

## Memory Bottlenecks

### Ranked by Typical Size (Large-Scale Training)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 🥇 ACTIVATIONS                                           │
│    Size: 1-10 GB (40-80% of total memory!)                 │
│    Scales with: batch_size, depth, spatial resolution       │
│    Bottleneck: Usually the limiting factor                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. 🥈 INPUT BATCH                                           │
│    Size: 100-500 MB                                         │
│    Scales with: batch_size, image resolution                │
│    Bottleneck: Significant for high-resolution images       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. 🥉 OPTIMIZER STATES (Adam)                               │
│    Size: 100-500 MB                                         │
│    Scales with: number of parameters                        │
│    Bottleneck: Can be significant for large models          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 4. PARAMETERS                                               │
│    Size: 50-200 MB (standard), ~0.1 MB (HyperNetwork!)     │
│    Scales with: model size                                  │
│    Bottleneck: Not usually, except for giant models         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 5. GRADIENTS                                                │
│    Size: Same as parameters                                 │
│    Scales with: model size                                  │
│    Bottleneck: Not usually                                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Observations

1. **Activations dominate:** Even with HyperNetworks' 440× parameter reduction, activations are still the largest component.

2. **Batch size is critical:** Doubling batch size approximately doubles activation memory.

3. **Early layers are expensive:** High-resolution feature maps in early layers consume significant memory.

4. **Deep networks accumulate:** More layers = more activations to store.

---

## HyperNetwork-Specific Considerations

### Memory Comparison: Standard vs. HyperNetwork

| Component | Standard ResNet-18 | HyperNetwork ResNet-18 | Savings |
|-----------|-------------------:|----------------------:|--------:|
| **Parameters** | 44 MB | 0.1 MB | **440×** ✓ |
| **Gradients** | 44 MB | 0.1 MB | **440×** ✓ |
| **Optimizer (Adam)** | 88 MB | 0.2 MB | **440×** ✓ |
| **Activations** | ~100 MB | ~100 MB | **1×** (same!) |
| **Generated Weights** | 0 MB | 5 MB | **overhead** |
| **Input Batch** | 1.5 MB | 1.5 MB | **1×** (same!) |
| **TOTAL** | ~280 MB | ~110 MB | **2.5×** |

### Key Insights

1. **Massive parameter savings (440×)** but activations unchanged.
2. **Total memory savings: ~2.5×** (much less than parameter compression suggests).
3. **Generated weights add overhead** (~5 MB) not present in standard networks.
4. **Training memory savings > inference savings** (no gradients/optimizer in inference).

### Where HyperNetworks Win

**Storage (disk/transmission):**
```
Standard ResNet-18 checkpoint: 44 MB (parameters only)
HyperNetwork ResNet-18 checkpoint: 0.1 MB (parameters only)
Compression ratio: 440× ✓✓✓
```

**Inference on memory-constrained devices:**
```
No gradients, no optimizer states needed
Only parameters + activations (single batch)
HyperNetwork advantage: ~50× smaller footprint
```

**Where HyperNetworks Don't Help Much:**

**Training on GPUs:**
```
Activations still dominate
Memory savings: only 2-3×, not 440×
```

---

## Memory Optimization Techniques

### 1. **Reduce Batch Size**

**Effect:** Linear reduction in activation memory

```python
# Before: batch_size = 256, activations = 2 GB
# After:  batch_size = 128, activations = 1 GB (2× less)

trainloader = DataLoader(dataset, batch_size=128)  # Reduced from 256
```

**Trade-off:**
- ✓ Reduced memory
- ✗ More iterations per epoch (slower)
- ✗ Less stable gradients (smaller batch = noisier estimates)

**Solution:** Use gradient accumulation to simulate large batches

---

### 2. **Gradient Accumulation**

**Idea:** Simulate large batch size by accumulating gradients over multiple small batches.

```python
accumulation_steps = 4  # Simulate 4× larger batch

optimizer.zero_grad()
for i, (inputs, labels) in enumerate(trainloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients (don't update yet)

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update parameters
        optimizer.zero_grad()  # Clear accumulated gradients
```

**Effect:**
- Effective batch size = batch_size × accumulation_steps
- Memory usage = only one small batch at a time

**Trade-off:**
- ✓ Large effective batch size with small memory
- ✗ Slower (more backward passes per update)

---

### 3. **Gradient Checkpointing**

**Idea:** Don't store all activations. Recompute them during backward pass.

```python
from torch.utils.checkpoint import checkpoint

class MyNetwork(nn.Module):
    def forward(self, x):
        # Checkpoint expensive blocks
        x = checkpoint(self.expensive_block1, x)
        x = checkpoint(self.expensive_block2, x)
        return self.final_layer(x)
```

**Effect:**
- Activation memory: ~50% reduction
- Computation: ~30% slower (recompute during backward)

**Trade-off:**
- ✓ Significant memory savings
- ✗ Slower training (recomputation overhead)

**Best for:** Very deep networks, limited GPU memory

---

### 4. **Mixed Precision Training (FP16/BF16)**

**Idea:** Use 16-bit floats instead of 32-bit for most operations.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in trainloader:
    optimizer.zero_grad()

    with autocast():  # Operations in FP16
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()  # Scale gradients to prevent underflow
    scaler.step(optimizer)
    scaler.update()
```

**Effect:**
- Parameters: 4 bytes → 2 bytes (2× reduction)
- Gradients: 4 bytes → 2 bytes (2× reduction)
- Activations: 4 bytes → 2 bytes (2× reduction)
- **Total memory: ~2× reduction**

**Trade-off:**
- ✓ 2× memory savings
- ✓ Faster computation (on modern GPUs with Tensor Cores)
- ✗ Potential numerical instability (mitigated by gradient scaling)

**Best for:** Modern GPUs (Volta, Turing, Ampere, Hopper with Tensor Cores)

---

### 5. **Activation Recomputation (Selective)**

**Idea:** Choose which activations to recompute vs. store based on memory/compute trade-off.

```python
# Recompute cheap operations (ReLU, BatchNorm)
# Store expensive operations (Convolution, Matrix Multiply)
```

**Libraries:**
- DeepSpeed ZeRO (automated)
- FairScale (PyTorch)

**Effect:** Fine-grained control over memory/compute trade-off

---

### 6. **Reduce Image Resolution**

**Effect:** Quadratic reduction in activation memory (spatial dimensions squared)

```python
# Before: 224×224 images
# After:  112×112 images (4× less activation memory)

transform = transforms.Compose([
    transforms.Resize(112),  # Reduced from 224
    transforms.ToTensor(),
])
```

**Trade-off:**
- ✓ Significant memory savings (quadratic)
- ✗ Loss of fine-grained visual information
- ✗ Reduced accuracy for tasks requiring detail

---

### 7. **Model Architecture Changes**

#### **a) Fewer Layers (Reduce Depth)**
```python
# Use ResNet-18 instead of ResNet-50
# 18 layers vs. 50 layers = fewer activations to store
```

#### **b) Smaller Channels**
```python
# Reduce channel dimensions (e.g., 64 → 32)
# Trade-off: reduced representational capacity
```

#### **c) Early Downsampling**
```python
# Downsample spatial resolution earlier in network
# Reduces activation sizes in subsequent layers
```

---

### 8. **CPU Offloading**

**Idea:** Store some tensors on CPU RAM, move to GPU only when needed.

```python
# Move gradients to CPU after computation
# Move optimizer states to CPU between steps
```

**Effect:** Trades GPU memory for CPU-GPU bandwidth

**Libraries:**
- DeepSpeed ZeRO-Offload
- PyTorch FSDP with CPU offload

**Trade-off:**
- ✓ Enables training of very large models
- ✗ Slower (CPU-GPU transfer overhead)

---

### 9. **Use Inplace Operations**

**Idea:** Modify tensors in-place instead of creating new ones.

```python
# Not inplace (creates new tensor)
x = F.relu(x)

# Inplace (modifies x directly)
x = F.relu(x, inplace=True)
```

**Effect:** Reduces temporary tensor allocations

**Caution:** Can interfere with gradient computation if used incorrectly

---

### 10. **Clear Unused Tensors**

```python
# Explicitly delete large tensors when done
del large_tensor
torch.cuda.empty_cache()  # Release cached memory
```

**Note:** Rarely needed; PyTorch's automatic memory management is good.

---

## Practical Guidelines

### 1. **Diagnosing OOM Errors**

If training crashes with "CUDA out of memory":

```python
# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max:       {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

**Common causes:**
1. Batch size too large (most common)
2. Image resolution too high
3. Network too deep
4. Accumulating tensors in a loop (memory leak)

**Solutions in order:**
1. Reduce batch size
2. Enable mixed precision (FP16)
3. Enable gradient checkpointing
4. Reduce image resolution
5. Use gradient accumulation
6. Use a smaller model

---

### 2. **Optimal Batch Size Selection**

**Strategy:** Binary search for maximum batch size that fits in memory.

```python
batch_sizes = [256, 128, 64, 32, 16, 8]

for batch_size in batch_sizes:
    try:
        trainloader = DataLoader(dataset, batch_size=batch_size)
        train_one_epoch(model, trainloader)
        print(f"Batch size {batch_size} works!")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            continue
        else:
            raise e
```

**Rule of thumb:**
- Use the largest batch size that fits in memory
- Smaller batches = noisier gradients but better generalization
- Larger batches = more stable but may generalize worse

---

### 3. **Memory Profiling**

**PyTorch Profiler:**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA],
             profile_memory=True) as prof:
    outputs = model(inputs)
    loss.backward()

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```

**NVIDIA Tools:**
```bash
nvidia-smi  # Monitor GPU memory in real-time
watch -n 1 nvidia-smi  # Update every 1 second
```

---

### 4. **HyperNetwork-Specific Tips**

**a) Memory advantage is in inference and storage:**
```
Deploy on edge devices: HyperNetworks shine here
Training on GPUs: Advantage is moderate (~2-3×)
```

**b) Generated weights overhead:**
```
~5 MB for ResNet-18 (36 layers)
Negligible compared to activations
```

**c) Batch size still matters:**
```
Activations dominate regardless of parameter compression
Use same batch size strategies as standard networks
```

---

### 5. **Hardware Considerations**

| GPU | Memory | Suitable For |
|-----|--------|--------------|
| **RTX 3060** | 12 GB | Small models, moderate batches |
| **RTX 3080** | 10-12 GB | Medium models, moderate batches |
| **RTX 3090** | 24 GB | Large models, large batches |
| **A100** | 40/80 GB | Very large models, very large batches |
| **H100** | 80 GB | Cutting-edge large-scale training |

**For HyperNetworks:**
- Parameter savings help, but activations still constrain batch size
- Can train models with 10× more layers compared to standard networks (same parameter budget)
- Enables deployment on resource-constrained devices

---

## Summary

### Key Takeaways

1. **Activations dominate GPU memory** (40-80%), not parameters.
2. **HyperNetworks save parameters (440×)** but total training memory only ~2-3× less.
3. **Batch size is the primary control** for memory usage (linear scaling).
4. **Mixed precision (FP16) provides easy 2× memory savings** on modern GPUs.
5. **Gradient checkpointing trades compute for memory** (~30% slower, 50% less memory).
6. **HyperNetworks excel in storage and inference** on constrained devices, not training memory.

### Decision Matrix

**If you have limited GPU memory:**
1. Start: Reduce batch size
2. Easy: Enable mixed precision (FP16)
3. Moderate: Gradient accumulation
4. Advanced: Gradient checkpointing
5. Last resort: CPU offloading, reduce resolution, smaller model

**For HyperNetworks specifically:**
- Same optimization strategies apply
- Parameter savings help but don't eliminate activation bottleneck
- Best use case: deployment on edge devices, not training efficiency

---

## References

- **PyTorch Memory Management:** https://pytorch.org/docs/stable/notes/cuda.html
- **Mixed Precision Training:** https://pytorch.org/docs/stable/amp.html
- **Gradient Checkpointing:** https://pytorch.org/docs/stable/checkpoint.html
- **HyperNetworks Paper:** Ha et al., "HyperNetworks" (2016)
- **Memory-Efficient Training:** https://huggingface.co/docs/transformers/perf_train_gpu_one

---

*Last Updated: 2025-01-XX*
