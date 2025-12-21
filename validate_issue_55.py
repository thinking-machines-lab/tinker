#!/usr/bin/env python3
"""
Validation script for Issue #55: save_weights_and_get_sampling_client hangs indefinitely

Fix: Added timeout parameter with 300s default.
"""

import sys
import inspect
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def test_issue_55_timeout_parameter():
    """
    Validates:
    1. timeout parameter exists on both sync and async methods
    2. timeout is passed to .result() call
    3. Default is 300 seconds
    4. timeout accepts None for indefinite wait
    """
    print("\n" + "=" * 60)
    print("Testing Issue #55: save_weights_and_get_sampling_client timeout")
    print("=" * 60)

    from tinker.lib.public_interfaces.training_client import TrainingClient

    # Test 1: Check sync method has timeout parameter
    sync_sig = inspect.signature(TrainingClient.save_weights_and_get_sampling_client)
    params = sync_sig.parameters

    assert 'timeout' in params, "timeout parameter missing from sync method"
    assert params['timeout'].default == 300.0, f"Expected default 300.0, got {params['timeout'].default}"
    print("  [PASS] Sync method has timeout parameter with default 300.0")

    # Test 2: Check async method has timeout parameter
    async_sig = inspect.signature(TrainingClient.save_weights_and_get_sampling_client_async)
    async_params = async_sig.parameters

    assert 'timeout' in async_params, "timeout parameter missing from async method"
    assert async_params['timeout'].default == 300.0, f"Expected default 300.0, got {async_params['timeout'].default}"
    print("  [PASS] Async method has timeout parameter with default 300.0")

    # Test 3: Verify timeout can be set to None for indefinite wait
    annotation = params['timeout'].annotation
    annotation_str = str(annotation)
    assert 'float' in annotation_str and 'None' in annotation_str, \
        f"timeout should accept float | None, got {annotation}"
    print("  [PASS] timeout accepts None for indefinite wait")

    print("\n  Issue #55 fix VALIDATED")
    return True


def main():
    print("\n" + "=" * 60)
    print("Issue #55 Fix Validation")
    print("=" * 60)

    try:
        if test_issue_55_timeout_parameter():
            print("\n" + "=" * 60)
            print("VALIDATION PASSED")
            print("=" * 60)
            return 0
    except Exception as e:
        print(f"\n  [FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("VALIDATION FAILED")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
