import torch
import sys
import os

# Add the parent directory to system path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from era_v3_session_6 import Net  # assuming your model class is named Net

def test_parameter_count():
    """Test if model has less than 20000 parameters"""
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_batch_normalization():
    """Test if model uses Batch Normalization"""
    model = Net()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"

def test_dropout():
    """Test if model uses Dropout"""
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_or_fc():
    """Test if model uses either GAP or Fully Connected layer"""
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_gap or has_fc, "Model should use either GAP or Fully Connected layer"

def test_accuracy():
    """Test if model achieves accuracy above 99.4%"""
    try:
        with open('final_accuracy.txt', 'r') as f:
            accuracy = float(f.read().strip())
        assert accuracy > 99.4, f"Model accuracy {accuracy}% should be greater than 99.4%"
    except FileNotFoundError:
        assert False, "final_accuracy.txt file not found. Make sure the notebook saves the final accuracy." 