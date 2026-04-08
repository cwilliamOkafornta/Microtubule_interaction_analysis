import os
import numpy as np
import pytest
from mt_interaction_core import MicrotubuleSpatialGraph

def test_amira_spatialgraph_io_fidelity(tmp_path):
    # Create a synthetic graph
    graph = MicrotubuleSpatialGraph()
    pts = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)
    seg = {
        'segment_id': 0,
        'mt_class': 1,
        'node1_pos': pts[0],
        'node2_pos': pts[1],
        'points': pts
    }
    graph.segments = [seg]
    
    # Save to temp file
    output_path = os.path.join(tmp_path, "test_out.am")
    graph.save_as_am(output_path)
    
    # Verify file exists and has content
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    # Reload and verify
    new_graph = MicrotubuleSpatialGraph()
    new_graph.load_from_am(output_path)
    
    assert len(new_graph.segments) == 1
    assert new_graph.segments[0]['mt_class'] == 1
    assert np.allclose(new_graph.segments[0]['points'], pts)

def test_amira_vertexset_io(tmp_path):
    graph = MicrotubuleSpatialGraph()
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    values = np.array([0.5, 0.9], dtype=np.float32)
    
    output_path = os.path.join(tmp_path, "test_points.am")
    graph.save_points_as_am(output_path, points, values=values, label="TestVal")
    
    assert os.path.exists(output_path)
    with open(output_path, 'rb') as f:
        content = f.read().decode('ascii', errors='ignore')
        assert "ContentType \"HxSpatialGraph\"" in content
        assert "VERTEX { float TestVal } @2" in content
