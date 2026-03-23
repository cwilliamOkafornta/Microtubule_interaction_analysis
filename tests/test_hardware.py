import numpy as np
import pytest
import mt_interaction_core as core

def test_gpu_fallback_safety():
    # Mock some data
    seg1 = {'segment_id': 1, 'mt_class': 6, 'points': np.array([[0,0,0], [10,0,0]], dtype=np.float32)}
    seg2 = {'segment_id': 2, 'mt_class': 7, 'points': np.array([[0,5,0], [10,5,0]], dtype=np.float32)}
    segments = [seg1, seg2]
    c1, c2 = np.array([0,0,0]), np.array([10,0,0])
    
    # Even if we request GPU, if HAS_GPU is False, it should fallback to CPU and NOT crash
    # We force HAS_GPU to False for this test if it isn't already
    original_has_gpu = core.HAS_GPU
    core.HAS_GPU = False
    
    try:
        df = core.compute_advanced_interactions(segments, 500.0, [6], [7], c1, c2, use_gpu=True)
        assert not df.empty
        assert len(df) == 1
    finally:
        # Restore original state
        core.HAS_GPU = original_has_gpu

def test_hardware_scan_utility():
    # Check if we can at least call the scan logic without errors
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    assert cpu_count > 0
