"""
Integration tests for Dynamic HyperNetwork implementation.
Tests end-to-end functionality of all components.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_1_imports():
    """Test 1: Verify all modules can be imported."""
    print("\n" + "="*60)
    print("Test 1: Import Validation")
    print("="*60)

    try:
        from src.puzzle_embedding import PuzzleEmbedding, HierarchicalPuzzleEmbedding
        from src.weight_generators import MultiComponentGenerator
        from src.dynamic_hypernetwork import DynamicHyperNetwork
        from src.arc_hypercompressor import (
            HyperCompressorFactory,
            TaskMetadataExtractor,
            HyperCompressorCheckpoint
        )
        from src.train_hypernetwork import HyperNetworkMetaTrainer
        from pre_processing import preprocess_tasks
        
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_2_task_preprocessing():
    """Test 2: Task preprocessing works correctly."""
    print("\n" + "="*60)
    print("Test 2: Task Preprocessing")
    print("="*60)

    try:
        from pre_processing import preprocess_tasks

        # Try to load a few tasks
        tasks = preprocess_tasks('training', ['00d62c1b', '0520fde7'])
        
        print(f"  ✓ Loaded {len(tasks)} tasks")
        
        # Verify task structure
        task = tasks[0]
        assert hasattr(task, 'task_name')
        assert hasattr(task, 'n_examples')
        assert hasattr(task, 'n_colors')
        assert hasattr(task, 'n_x')
        assert hasattr(task, 'n_y')
        assert hasattr(task, 'multitensor_system')
        
        print(f"  ✓ Task '{task.task_name}' structure valid")
        print(f"    - Examples: {task.n_examples}")
        print(f"    - Colors: {task.n_colors}")
        print(f"    - Grid: {task.n_x}×{task.n_y}")
        
        return True
    except Exception as e:
        print(f"  ✗ Task preprocessing failed: {e}")
        return False


def test_3_model_initialization():
    """Test 3: Model can be initialized with default config."""
    print("\n" + "="*60)
    print("Test 3: Model Initialization")
    print("="*60)

    try:
        from src.dynamic_hypernetwork import DynamicHyperNetwork

        model = DynamicHyperNetwork(
            num_puzzles=100,
            z_dim=64,
            n_layers=4,
            use_metadata=True
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model initialized")
        print(f"    - Total parameters: {num_params:,}")
        print(f"    - Embedding dimension: 64")
        print(f"    - Number of layers: 4")
        
        return True
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        return False


def test_4_forward_pass():
    """Test 4: Forward pass generates weights correctly."""
    print("\n" + "="*60)
    print("Test 4: Forward Pass")
    print("="*60)

    try:
        from src.dynamic_hypernetwork import DynamicHyperNetwork

        model = DynamicHyperNetwork(
            num_puzzles=100,
            z_dim=64,
            n_layers=4,
            use_metadata=True
        )
        
        # Test metadata
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
        
        # Forward pass
        embedding, weights = model(
            task_name="test_task",
            task_metadata=task_metadata
        )
        
        print(f"  ✓ Forward pass successful")
        print(f"    - Embedding shape: {embedding.shape}")
        print(f"    - Generated {len(weights)} weight components")
        print(f"    - Components: {list(weights.keys())}")
        
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False


def test_5_training_step():
    """Test 5: Training step with optimizer."""
    print("\n" + "="*60)
    print("Test 5: Training Step")
    print("="*60)

    try:
        from src.dynamic_hypernetwork import DynamicHyperNetwork

        model = DynamicHyperNetwork(
            num_puzzles=100,
            z_dim=32,  # Smaller for speed
            n_layers=2,
            use_metadata=True
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
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
        
        # Training step
        optimizer.zero_grad()
        embedding, weights = model("test_task", task_metadata)
        
        # Dummy loss (just for testing gradient flow)
        loss = embedding.abs().mean()
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training step successful")
        print(f"    - Loss: {loss.item():.4f}")
        print(f"    - Gradients computed")
        print(f"    - Optimizer step completed")
        
        return True
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        return False


def test_6_factory_presets():
    """Test 6: Factory presets work correctly."""
    print("\n" + "="*60)
    print("Test 6: Factory Presets")
    print("="*60)

    try:
        from src.arc_hypercompressor import HyperCompressorFactory

        # Test all presets
        lightweight = HyperCompressorFactory.create_lightweight('cpu')
        default = HyperCompressorFactory.create_default('cpu')
        large = HyperCompressorFactory.create_large('cpu')
        
        lw_params = sum(p.numel() for p in lightweight.parameters())
        def_params = sum(p.numel() for p in default.parameters())
        lg_params = sum(p.numel() for p in large.parameters())
        
        print(f"  ✓ All factory presets created")
        print(f"    - Lightweight: {lw_params:,} parameters")
        print(f"    - Default: {def_params:,} parameters")
        print(f"    - Large: {lg_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Factory presets failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("Dynamic HyperNetwork - Integration Tests")
    print("="*60)

    tests = [
        test_1_imports,
        test_2_task_preprocessing,
        test_3_model_initialization,
        test_4_forward_pass,
        test_5_training_step,
        test_6_factory_presets,
    ]

    results = []
    for i, test in enumerate(tests, 1):
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test {i} crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, result in enumerate(results, 1):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  Test {i}: {status}")
    
    print("\n" + "-"*60)
    print(f"  Total: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print("="*60 + "\n")
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
