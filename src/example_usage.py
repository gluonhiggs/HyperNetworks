"""
Example usage scripts for Dynamic HyperNetwork.
Demonstrates different usage patterns and configurations.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamic_hypernetwork import DynamicHyperNetwork
from src.arc_hypercompressor import (
    HyperCompressorFactory,
    TaskMetadataExtractor,
    HyperCompressorCheckpoint
)
from pre_processing import preprocess_tasks


def example_1_basic_usage():
    """
    Example 1: Basic hypernetwork creation and weight generation.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # Create hypernetwork
    model = DynamicHyperNetwork(
        num_puzzles=1000,
        z_dim=64,
        n_layers=4,
        use_metadata=True
    )
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"✓ Created hypernetwork with {sum(p.numel() for p in model.parameters())} parameters")

    # Generate weights for a task
    task_name = "00d62c1b"
    task_metadata = {
        'n_examples': 4,
        'n_colors': 3,
        'n_x': 10,
        'n_y': 10,
        'n_train': 3,
        'n_test': 1,
        'in_out_same_size': False,
        'all_in_same_size': False,
    }

    # Get task embedding
    embedding = model.get_task_embedding(task_name, task_metadata)
    print(f"✓ Task embedding shape: {embedding.shape}")

    # Generate weights
    weights = model.generate_weights(task_name, task_metadata)
    print(f"✓ Generated weights for {len(weights)} components")

    return model, weights


def example_2_factory_presets():
    """
    Example 2: Using factory presets for different configurations.
    """
    print("\n" + "="*60)
    print("Example 2: Factory Presets")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Lightweight model
    lightweight = HyperCompressorFactory.create_lightweight(device)
    print(f"✓ Lightweight model: {sum(p.numel() for p in lightweight.parameters())} params")

    # Default model
    default = HyperCompressorFactory.create_default(device)
    print(f"✓ Default model: {sum(p.numel() for p in default.parameters())} params")

    # Large model
    large = HyperCompressorFactory.create_large(device)
    print(f"✓ Large model: {sum(p.numel() for p in large.parameters())} params")

    return lightweight, default, large


def example_3_weight_caching():
    """
    Example 3: Using weight caching for inference speedup.
    """
    print("\n" + "="*60)
    print("Example 3: Weight Caching")
    print("="*60)

    model = HyperCompressorFactory.create_default('cpu')

    # Without caching
    import time
    start = time.time()
    for i in range(100):
        weights = model.generate_weights(f"task_{i % 10}")
    no_cache_time = time.time() - start
    print(f"✓ 100 generations without cache: {no_cache_time:.3f}s")

    # With caching
    model.enable_cache()
    start = time.time()
    for i in range(100):
        weights = model.generate_weights(f"task_{i % 10}", use_cache=True)
    cache_time = time.time() - start
    print(f"✓ 100 generations with cache: {cache_time:.3f}s")
    print(f"✓ Speedup: {no_cache_time / cache_time:.2f}x")

    return model


def example_4_checkpoint_save_load():
    """
    Example 4: Saving and loading model checkpoints.
    """
    print("\n" + "="*60)
    print("Example 4: Checkpoint Save/Load")
    print("="*60)

    # Create and save model
    model = HyperCompressorFactory.create_default('cpu')
    checkpoint_path = '/tmp/hypernet_test.pt'

    HyperCompressorCheckpoint.save(
        model,
        checkpoint_path,
        metadata={'experiment': 'test', 'version': '1.0'}
    )
    print(f"✓ Model saved to {checkpoint_path}")

    # Load model
    loaded_model = HyperCompressorCheckpoint.load(checkpoint_path, device='cpu')
    print(f"✓ Model loaded from {checkpoint_path}")

    # Verify weights match
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2), "Loaded weights don't match!"
    print("✓ Checkpoint integrity verified")

    return model, loaded_model


def example_5_task_metadata_extraction():
    """
    Example 5: Extracting metadata from ARC Task objects.
    """
    print("\n" + "="*60)
    print("Example 5: Task Metadata Extraction")
    print("="*60)

    # Load a real ARC task
    try:
        tasks = preprocess_tasks('training', ['00d62c1b'])
        task = tasks[0]

        # Extract metadata
        metadata = TaskMetadataExtractor.extract(task)
        print(f"✓ Extracted metadata for task '{task.task_name}':")
        for key, value in metadata.items():
            print(f"    {key}: {value}")

        return task, metadata

    except Exception as e:
        print(f"  ⚠ Could not load ARC tasks: {e}")
        print("  (This example requires ARC dataset in 'dataset/' directory)")
        return None, None


def example_6_custom_configuration():
    """
    Example 6: Creating custom hypernetwork configurations.
    """
    print("\n" + "="*60)
    print("Example 6: Custom Configuration")
    print("="*60)

    # Custom channel dimensions
    custom_dims = {
        'share_up_dim': 32,      # Doubled from default 16
        'share_down_dim': 16,    # Doubled from default 8
        'decoding_dim': 8,       # Doubled from default 4
        'softmax_dim': 4,        # Doubled from default 2
        'cummax_dim': 8,         # Doubled from default 4
        'shift_dim': 8,          # Doubled from default 4
        'nonlinear_dim': 32,     # Doubled from default 16
    }

    model = DynamicHyperNetwork(
        num_puzzles=1500,
        z_dim=96,
        n_layers=5,
        use_metadata=True,
        use_hierarchical=True,
        channel_dims=custom_dims
    )

    print(f"✓ Created custom hypernetwork:")
    print(f"    Puzzles: 1500")
    print(f"    Embedding dim: 96")
    print(f"    Layers: 5")
    print(f"    Hierarchical: True")
    print(f"    Total params: {sum(p.numel() for p in model.parameters())}")

    return model


def run_all_examples():
    """Run all examples sequentially."""
    print("\n" + "="*60)
    print("Dynamic HyperNetwork - Example Usage")
    print("="*60)

    try:
        example_1_basic_usage()
    except Exception as e:
        print(f"  ✗ Example 1 failed: {e}")

    try:
        example_2_factory_presets()
    except Exception as e:
        print(f"  ✗ Example 2 failed: {e}")

    try:
        example_3_weight_caching()
    except Exception as e:
        print(f"  ✗ Example 3 failed: {e}")

    try:
        example_4_checkpoint_save_load()
    except Exception as e:
        print(f"  ✗ Example 4 failed: {e}")

    try:
        example_5_task_metadata_extraction()
    except Exception as e:
        print(f"  ✗ Example 5 failed: {e}")

    try:
        example_6_custom_configuration()
    except Exception as e:
        print(f"  ✗ Example 6 failed: {e}")

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_examples()
