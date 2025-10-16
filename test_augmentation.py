"""
Quick test to verify augmentation logic works correctly.
"""
import sys
sys.path.insert(0, '.')

from run_arc_hyper_augmented import (
    parse_augmented_task_name,
    load_tasks_with_augmentation
)

print("Testing parse_augmented_task_name()...")
test_cases = [
    ('c9680e90', ('c9680e90', 'original')),
    ('c9680e90_rot90', ('c9680e90', 'rot90')),
    ('c9680e90_rot180', ('c9680e90', 'rot180')),
    ('c9680e90_flip_h', ('c9680e90', 'flip_h')),
    ('c9680e90_flip_v', ('c9680e90', 'flip_v')),
]

for input_name, expected in test_cases:
    result = parse_augmented_task_name(input_name)
    status = "✓" if result == expected else "✗"
    print(f"  {status} {input_name:20s} -> {result}")
    if result != expected:
        print(f"     Expected: {expected}")

print("\nTesting load_tasks_with_augmentation()...")
try:
    import os
    os.environ['ARC_DATA_DIR'] = 'dataset/'

    # Test with small subset
    augmented_tasks = load_tasks_with_augmentation(
        'training',
        augmentation_types=['original', 'rot90', 'rot180']
    )

    print(f"  ✓ Loaded {len(augmented_tasks)} augmented tasks")
    print(f"  Example tasks:")
    for i, task in enumerate(augmented_tasks[:9]):
        print(f"    {task}")
        if i >= 8:
            break

    # Verify format
    original_count = sum(1 for t in augmented_tasks if '_' not in t)
    augmented_count = len(augmented_tasks) - original_count
    print(f"\n  ✓ Original tasks: {original_count}")
    print(f"  ✓ Augmented tasks: {augmented_count}")
    print(f"  ✓ Total: {len(augmented_tasks)}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
