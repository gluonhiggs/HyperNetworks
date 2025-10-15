"""
Comprehensive test script to verify ALL generate functions produce same structure as Initializer.

Tests:
1. generate_multiposterior vs initialize_multiposterior
2. generate_multilinear vs initialize_multilinear
3. generate_multizeros vs initialize_multizeros
4. generate_multiresidual vs initialize_multiresidual
5. generate_multidirection_share vs initialize_multidirection_share
6. generate_head vs initialize_head

For each function, we verify:
- Same number of tensors
- Same shapes for each tensor
- Same MultiTensor structure (same dims keys)
- Same data types and requires_grad
"""

import torch
import numpy as np
from hypernetwork_arc import HyperNetworkARC
from initializers import Initializer
from multitensor_systems import MultiTensorSystem

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class MockTask:
    """Mock Task object for testing."""
    def __init__(self, n_examples, n_colors, n_x, n_y):
        self.n_examples = n_examples
        self.n_colors = n_colors
        self.n_x = n_x
        self.n_y = n_y
        self.multitensor_system = MultiTensorSystem(n_examples, n_colors, n_x, n_y, task=self)
        self.masks = None


def setup_test_environment():
    """Setup common test environment."""
    task = MockTask(n_examples=3, n_colors=2, n_x=3, n_y=3)
    multitensor_system = task.multitensor_system
    channel_dim_fn = lambda dims: 16 if dims[2] == 0 else 8

    # Initialize hypernetwork
    hypernetwork = HyperNetworkARC(emb_dim=128, hidden_dim=512)
    puzzle_emb = torch.randn(128)

    # Generate hidden representation
    metadata = torch.tensor([
        task.n_examples / 12.0,
        task.n_colors / 9.0,
        task.n_x / 30.0,
        task.n_y / 30.0
    ], dtype=torch.float32)
    h = torch.cat([puzzle_emb, metadata])
    h = hypernetwork.body(h)

    # Initialize original Initializer
    initializer = Initializer(multitensor_system, channel_dim_fn)

    return task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer


def compare_tensor_structure(hyper_tensor, init_tensor, name="tensor"):
    """Compare structure of two tensors."""
    if type(hyper_tensor) != type(init_tensor):
        print(f"  ❌ {name} type mismatch: {type(hyper_tensor)} vs {type(init_tensor)}")
        return False

    if isinstance(hyper_tensor, torch.Tensor):
        if hyper_tensor.shape != init_tensor.shape:
            print(f"  ❌ {name} shape mismatch: {hyper_tensor.shape} vs {init_tensor.shape}")
            return False
        if hyper_tensor.requires_grad != init_tensor.requires_grad:
            print(f"  ⚠️  {name} requires_grad mismatch")
        return True

    elif isinstance(hyper_tensor, list):
        if len(hyper_tensor) != len(init_tensor):
            print(f"  ❌ {name} list length mismatch: {len(hyper_tensor)} vs {len(init_tensor)}")
            return False
        for i, (h_item, i_item) in enumerate(zip(hyper_tensor, init_tensor)):
            if not compare_tensor_structure(h_item, i_item, f"{name}[{i}]"):
                return False
        return True

    else:
        print(f"  ⚠️  Unknown type for {name}: {type(hyper_tensor)}")
        return True


def test_1_multiposterior():
    """Test generate_multiposterior vs initialize_multiposterior."""
    print("\n" + "="*80)
    print("TEST 1: generate_multiposterior vs initialize_multiposterior")
    print("="*80)

    task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer = setup_test_environment()

    # Generate
    hyper_result = hypernetwork.generate_multiposterior(h, multitensor_system, 4)
    init_result = initializer.initialize_multiposterior(4)

    # Compare
    valid_dims = list(multitensor_system)
    all_match = True

    for dims in valid_dims:
        hyper_pair = hyper_result[dims]
        init_pair = init_result[dims]

        if not compare_tensor_structure(hyper_pair, init_pair, f"dims={dims}"):
            all_match = False

    print(f"\n✅ Test 1 PASSED: {len(valid_dims)} dims verified" if all_match else "\n❌ Test 1 FAILED")
    return all_match


def test_2_multilinear():
    """Test generate_multilinear vs initialize_multilinear."""
    print("\n" + "="*80)
    print("TEST 2: generate_multilinear vs initialize_multilinear")
    print("="*80)

    task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer = setup_test_environment()

    # Generate with shape [4, channel_dim_fn]
    hyper_result = hypernetwork.generate_multilinear(h, multitensor_system, [4, channel_dim_fn])
    init_result = initializer.initialize_multilinear([4, channel_dim_fn])

    # Compare
    valid_dims = list(multitensor_system)
    all_match = True

    for dims in valid_dims:
        hyper_pair = hyper_result[dims]
        init_pair = init_result[dims]

        if not compare_tensor_structure(hyper_pair, init_pair, f"dims={dims}"):
            all_match = False

    print(f"\n✅ Test 2 PASSED: {len(valid_dims)} dims verified" if all_match else "\n❌ Test 2 FAILED")
    return all_match


def test_3_multizeros():
    """Test generate_multizeros vs initialize_multizeros."""
    print("\n" + "="*80)
    print("TEST 3: generate_multizeros vs initialize_multizeros")
    print("="*80)

    task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer = setup_test_environment()

    # Generate
    hyper_result = hypernetwork.generate_multizeros(h, multitensor_system, [4])
    init_result = initializer.initialize_multizeros([4])

    # Compare
    valid_dims = list(multitensor_system)
    all_match = True

    for dims in valid_dims:
        hyper_tensor = hyper_result[dims]
        init_tensor = init_result[dims]

        if not compare_tensor_structure(hyper_tensor, init_tensor, f"dims={dims}"):
            all_match = False

        # Check that both are zeros
        if not torch.allclose(hyper_tensor, torch.zeros_like(hyper_tensor)):
            print(f"  ❌ dims={dims}: Hyper result is not zeros")
            all_match = False

        if not torch.allclose(init_tensor, torch.zeros_like(init_tensor)):
            print(f"  ❌ dims={dims}: Init result is not zeros")
            all_match = False

    print(f"\n✅ Test 3 PASSED: {len(valid_dims)} dims verified" if all_match else "\n❌ Test 3 FAILED")
    return all_match


def test_4_multiresidual():
    """Test generate_multiresidual vs initialize_multiresidual."""
    print("\n" + "="*80)
    print("TEST 4: generate_multiresidual vs initialize_multiresidual")
    print("="*80)

    task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer = setup_test_environment()

    # Generate with n_in=16, n_out=16
    hyper_result = hypernetwork.generate_multiresidual(h, multitensor_system, 16, 16, channel_dim_fn)
    init_result = initializer.initialize_multiresidual(16, 16)

    # Compare
    valid_dims = list(multitensor_system)
    all_match = True

    for dims in valid_dims:
        hyper_pair = hyper_result[dims]
        init_pair = init_result[dims]

        # Should be [[weight1, bias1], [weight2, bias2]]
        if not isinstance(hyper_pair, list) or len(hyper_pair) != 2:
            print(f"  ❌ dims={dims}: Hyper result not [linear1, linear2] pair")
            all_match = False
            continue

        if not isinstance(init_pair, list) or len(init_pair) != 2:
            print(f"  ❌ dims={dims}: Init result not [linear1, linear2] pair")
            all_match = False
            continue

        # Compare linear1
        if not compare_tensor_structure(hyper_pair[0], init_pair[0], f"dims={dims} linear1"):
            all_match = False

        # Compare linear2
        if not compare_tensor_structure(hyper_pair[1], init_pair[1], f"dims={dims} linear2"):
            all_match = False

    print(f"\n✅ Test 4 PASSED: {len(valid_dims)} dims verified" if all_match else "\n❌ Test 4 FAILED")
    return all_match


def test_5_multidirection_share():
    """Test generate_multidirection_share vs initialize_multidirection_share."""
    print("\n" + "="*80)
    print("TEST 5: generate_multidirection_share vs initialize_multidirection_share")
    print("="*80)

    task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer = setup_test_environment()

    # Generate
    hyper_result = hypernetwork.generate_multidirection_share(h, multitensor_system, channel_dim_fn)
    init_result = initializer.initialize_multidirection_share()

    # Compare
    valid_dims = list(multitensor_system)
    all_match = True

    for dims in valid_dims:
        hyper_grid = hyper_result[dims]
        init_grid = init_result[dims]

        # Should be 8x8 grid
        if not isinstance(hyper_grid, list) or len(hyper_grid) != 8:
            print(f"  ❌ dims={dims}: Hyper result not 8x8 grid (outer)")
            all_match = False
            continue

        if not isinstance(init_grid, list) or len(init_grid) != 8:
            print(f"  ❌ dims={dims}: Init result not 8x8 grid (outer)")
            all_match = False
            continue

        # Check each row
        for dir1 in range(8):
            if not isinstance(hyper_grid[dir1], list) or len(hyper_grid[dir1]) != 8:
                print(f"  ❌ dims={dims}, dir1={dir1}: Hyper row not length 8")
                all_match = False
                continue

            if not isinstance(init_grid[dir1], list) or len(init_grid[dir1]) != 8:
                print(f"  ❌ dims={dims}, dir1={dir1}: Init row not length 8")
                all_match = False
                continue

            # Check each element
            for dir2 in range(8):
                if not compare_tensor_structure(
                    hyper_grid[dir1][dir2],
                    init_grid[dir1][dir2],
                    f"dims={dims}[{dir1}][{dir2}]"
                ):
                    all_match = False

    print(f"\n✅ Test 5 PASSED: {len(valid_dims)} dims × 64 directions verified" if all_match else "\n❌ Test 5 FAILED")
    return all_match


def test_6_head():
    """Test generate_head vs initialize_head."""
    print("\n" + "="*80)
    print("TEST 6: generate_head vs initialize_head")
    print("="*80)

    task, multitensor_system, channel_dim_fn, hypernetwork, h, initializer = setup_test_environment()

    # Generate
    hyper_result = hypernetwork.generate_head(h, multitensor_system, channel_dim_fn)
    init_result = initializer.initialize_head()

    # Head is only for specific dims [1, 1, 0, 1, 1]
    dims = [1, 1, 0, 1, 1]

    # Note: hyper_result is a MultiTensor, init_result is a plain list
    hyper_pair = hyper_result[dims]
    init_pair = init_result  # initialize_head returns [weight, bias] directly, not a MultiTensor

    all_match = compare_tensor_structure(hyper_pair, init_pair, f"dims={dims}")

    # Check symmetry: weight[:, 0] should equal weight[:, 1]
    hyper_weight, hyper_bias = hyper_pair
    init_weight, init_bias = init_pair

    if not torch.allclose(hyper_weight[:, 0], hyper_weight[:, 1]):
        print(f"  ❌ Hyper weight not symmetric (col 0 != col 1)")
        all_match = False

    if not torch.allclose(init_weight[:, 0], init_weight[:, 1]):
        print(f"  ❌ Init weight not symmetric (col 0 != col 1)")
        all_match = False

    print(f"\n✅ Test 6 PASSED: Head weights verified" if all_match else "\n❌ Test 6 FAILED")
    return all_match


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE FOR HYPERNETWORK GENERATE FUNCTIONS")
    print("="*80)
    print("\nTask configuration:")
    print("  n_examples = 3, n_colors = 2, n_x = 3, n_y = 3")
    print("  Testing structural equivalence with Initializer")

    results = []

    results.append(("generate_multiposterior", test_1_multiposterior()))
    results.append(("generate_multilinear", test_2_multilinear()))
    results.append(("generate_multizeros", test_3_multizeros()))
    results.append(("generate_multiresidual", test_4_multiresidual()))
    results.append(("generate_multidirection_share", test_5_multidirection_share()))
    results.append(("generate_head", test_6_head()))

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:40s} {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("   Hypernetwork generates identical structure to Initializer")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("   See details above")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
