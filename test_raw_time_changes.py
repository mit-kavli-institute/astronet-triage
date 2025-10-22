#!/usr/bin/env python3
"""
Test script to verify that the raw time changes work correctly.
This script tests the key functions that were modified.
"""

import numpy as np
import sys
import os

# Add the astronet directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'astronet'))

from astronet.preprocess import preprocess

def test_align_raw_time():
    """Test the align_raw_time function"""
    print("Testing align_raw_time...")

    # Create test data
    detrended_time = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    detrended_flux = np.array([1.0, 0.99, 0.98, 0.99, 1.0, 0.99, 0.98, 0.99, 1.0])
    period = 2.0
    epoch = 1.0

    raw_time_aligned, raw_flux_aligned = preprocess.align_raw_time(
        detrended_time, detrended_flux, period, epoch)

    print(f"Input time shape: {detrended_time.shape}")
    print(f"Output raw_time shape: {raw_time_aligned.shape}")
    print(f"Output raw_flux shape: {raw_flux_aligned.shape}")

    # Check that lengths match
    assert len(raw_time_aligned) == len(detrended_time), "Length mismatch in align_raw_time"
    assert len(raw_flux_aligned) == len(detrended_flux), "Length mismatch in align_raw_time"

    print("✓ align_raw_time test passed")
    return True

def test_view_functions_with_raw_time():
    """Test view functions with raw time parameters"""
    print("Testing view functions with raw time...")

    # Create test data
    tic = 12345
    time = np.linspace(-1, 1, 100)
    flux = 1.0 + 0.01 * np.sin(2 * np.pi * time) + 0.001 * np.random.randn(100)
    period = 2.0
    duration = 0.1

    # Create raw time data (simulate different cadence)
    raw_time = np.linspace(-1, 1, 200)  # Higher cadence
    raw_flux = 1.0 + 0.01 * np.sin(2 * np.pi * raw_time) + 0.001 * np.random.randn(200)

    try:
        # Test global_view with raw time
        view, std, mask, scale, depth = preprocess.global_view(
            tic, time, flux, period, all_30min=False, raw_time=raw_time, raw_flux=raw_flux)
        print(f"✓ global_view with raw_time: view shape {view.shape}")

        # Test local_view with raw time
        view, std, mask, scale, depth = preprocess.local_view(
            tic, time, flux, period, duration, all_30min=False, raw_time=raw_time, raw_flux=raw_flux)
        print(f"✓ local_view with raw_time: view shape {view.shape}")

        # Test secondary_view with raw time
        (view, std, mask, scale, depth), t0 = preprocess.secondary_view(
            tic, time, flux, period, duration, all_30min=False, raw_time=raw_time, raw_flux=raw_flux)
        print(f"✓ secondary_view with raw_time: view shape {view.shape}, t0={t0:.3f}")

        # Test sample_segments_view with raw time
        fold_num = np.random.randint(0, 3, len(time))
        view = preprocess.sample_segments_view(
            tic, time, flux, fold_num, period, duration, all_30min=False, raw_time=raw_time, raw_flux=raw_flux)
        print(f"✓ sample_segments_view with raw_time: view shape {view.shape}")

        return True

    except Exception as e:
        print(f"✗ Error testing view functions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_view_functions_without_raw_time():
    """Test view functions without raw time (should still work)"""
    print("Testing view functions without raw time...")

    # Create test data
    tic = 12345
    time = np.linspace(-1, 1, 100)
    flux = 1.0 + 0.01 * np.sin(2 * np.pi * time) + 0.001 * np.random.randn(100)
    period = 2.0
    duration = 0.1

    try:
        # Test global_view without raw time
        view, std, mask, scale, depth = preprocess.global_view(
            tic, time, flux, period, all_30min=True)
        print(f"✓ global_view without raw_time: view shape {view.shape}")

        # Test local_view without raw time
        view, std, mask, scale, depth = preprocess.local_view(
            tic, time, flux, period, duration, all_30min=True)
        print(f"✓ local_view without raw_time: view shape {view.shape}")

        return True

    except Exception as e:
        print(f"✗ Error testing view functions without raw time: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing raw time changes")
    print("=" * 50)

    tests = [
        test_align_raw_time,
        test_view_functions_with_raw_time,
        test_view_functions_without_raw_time,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 50)

    if passed == total:
        print("🎉 All tests passed! The raw time changes are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
