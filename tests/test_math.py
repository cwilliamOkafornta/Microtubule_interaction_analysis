import numpy as np
import pytest
from mt_interaction_core import MicrotubuleSpatialGraph, approximate_direction, compute_advanced_interactions

def test_approximate_direction():
    # Test a simple line along X axis
    pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    dir_vec = approximate_direction(pts)
    assert np.allclose(dir_vec, [1, 0, 0])

    # Test a simple line along Z axis
    pts = np.array([[0, 0, 0], [0, 0, 5]], dtype=np.float32)
    dir_vec = approximate_direction(pts)
    assert np.allclose(dir_vec, [0, 0, 1])

def test_interaction_math_synthetic():
    # Create two parallel segments 200A apart
    # Segment 1: [0,0,0] to [1000,0,0] (Class 6)
    # Segment 2: [0,200,0] to [1000,200,0] (Class 7)
    
    seg1 = {
        'segment_id': 1,
        'mt_class': 6,
        'points': np.array([[x, 0, 0] for x in range(0, 1001, 100)], dtype=np.float32)
    }
    seg2 = {
        'segment_id': 2,
        'mt_class': 7,
        'points': np.array([[x, 200, 0] for x in range(0, 1001, 100)], dtype=np.float32)
    }
    
    segments = [seg1, seg2]
    dist_threshold = 500.0
    ref_class = [6]
    neighbor_class = [7]
    
    # Define arbitrary spindle centroids
    c1 = np.array([-100, 0, 0])
    c2 = np.array([1100, 0, 0])
    
    df = compute_advanced_interactions(segments, dist_threshold, ref_class, neighbor_class, c1, c2)
    
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]['Orientation'] == 'Parallel'
    assert df.iloc[0]['Mean_Dist_A'] == 200.0
    # Interaction length should be approx 1000
    assert 990 <= df.iloc[0]['Int_Length_A'] <= 1010
