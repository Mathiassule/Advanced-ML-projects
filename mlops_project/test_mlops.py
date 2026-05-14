import pandas as pd
import numpy as np

# --- UNIT TESTS FOR MLOPS PIPELINE ---
# In production, these tests run automatically on GitHub every time code is pushed.
# If they fail, the code is blocked from merging into the main product.

def test_reference_data_generation():
    """Test that the baseline training data generates correctly."""
    np.random.seed(42)
    n = 1000
    temp = np.random.normal(70, 10, n)
    
    # 1. Test data shape
    assert len(temp) == 1000, "Dataset did not generate the correct number of rows."
    
    # 2. Test statistical integrity (Mean should be close to 70)
    mean_temp = np.mean(temp)
    assert 68.0 < mean_temp < 72.0, f"Data anomaly detected! Mean temp was {mean_temp}"

def test_drift_detection_logic():
    """Test that the mathematical logic for detecting drift works."""
    # Simulate Reference Mean
    ref_mean = 70.0
    
    # Simulate Current Mean (Drifted)
    curr_mean = 75.0
    
    # Calculate % shift
    percent_shift = abs((curr_mean - ref_mean) / ref_mean) * 100
    is_drifted = percent_shift > 5.0
    
    # Assert that our logic correctly flags a 5% shift as True
    assert is_drifted == True, "Drift detection math failed to flag a >5% shift!"

def test_healthy_data_logic():
    """Test that healthy data does NOT trigger a false alarm."""
    ref_mean = 70.0
    curr_mean = 71.0 # Very small shift
    
    percent_shift = abs((curr_mean - ref_mean) / ref_mean) * 100
    is_drifted = percent_shift > 5.0
    
    # Assert that our logic correctly ignores minor fluctuations
    assert is_drifted == False, "False positive! Drift detected on healthy data."

if __name__ == "__main__":
    print("Run this file using: pytest test_mlops.py")