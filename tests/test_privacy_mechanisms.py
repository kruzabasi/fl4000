import pytest
import numpy as np
from typing import List, Optional, Tuple

# Assuming clipping logic is part of Client or a utility function
# Let's test a standalone clipping function for clarity
ModelParams = List[np.ndarray]

def _calculate_l2_norm(params: ModelParams) -> float:
    squared_norms = [np.sum(np.square(p)) for p in params]
    return np.sqrt(np.sum(squared_norms))

def clip_update_func(raw_update: ModelParams, clip_norm: Optional[float]) -> ModelParams:
    """Standalone clipping function for testing."""
    if clip_norm is None or clip_norm <= 0:
        return raw_update
    l2_norm = _calculate_l2_norm(raw_update)
    if l2_norm > 0:
        scale = min(1.0, clip_norm / l2_norm)
        return [layer * scale for layer in raw_update]
    else:
        return raw_update

@pytest.mark.parametrize("raw_update_list, clip_norm, expected_norm", [
    # Case 1: Norm below threshold
    ([np.array([1.0, 1.0]), np.array([1.0])], 3.0, np.sqrt(1**2+1**2+1**2)), # norm = sqrt(3) < 3
    # Case 2: Norm above threshold
    ([np.array([3.0, 4.0]), np.array([0.0])], 4.0, 4.0), # norm = sqrt(9+16) = 5 > 4
    # Case 3: Norm equals threshold
    ([np.array([3.0, 4.0]), np.array([0.0])], 5.0, 5.0), # norm = 5 == 5
    # Case 4: Zero norm
    ([np.array([0.0, 0.0]), np.array([0.0])], 5.0, 0.0),
    # Case 5: No clipping (clip_norm is None or 0)
    ([np.array([3.0, 4.0]), np.array([0.0])], None, 5.0),
    ([np.array([3.0, 4.0]), np.array([0.0])], 0.0, 5.0),
])
def test_l2_clipping(raw_update_list, clip_norm, expected_norm):
    """Tests L2 norm clipping logic."""
    clipped_update = clip_update_func(raw_update_list, clip_norm)
    output_norm = _calculate_l2_norm(clipped_update)

    assert len(clipped_update) == len(raw_update_list)
    # Allow for floating point inaccuracies
    assert abs(output_norm - expected_norm) < 1e-6

    # Check direction is preserved if norm > 0 and clipping occurred
    original_norm = _calculate_l2_norm(raw_update_list)
    if clip_norm is not None and clip_norm > 0 and original_norm > clip_norm and original_norm > 0:
        # Check if vectors are parallel (scaled version)
        scale = clip_norm / original_norm
        expected_after_clip = [layer * scale for layer in raw_update_list]
        for i in range(len(clipped_update)):
            np.testing.assert_allclose(clipped_update[i], expected_after_clip[i], rtol=1e-5)


def test_noise_addition_shape():
    """Tests noise addition preserves shape and changes values."""
    summed_update: ModelParams = [np.random.rand(10, 5), np.random.rand(5)]
    noise_std_dev = 0.1

    noisy_sum = []
    noise_added_check = False
    for layer_sum in summed_update:
        noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=layer_sum.shape)
        noisy_layer = layer_sum + noise
        noisy_sum.append(noisy_layer)
        # Check if noise actually changed the values (unlikely to be exactly zero)
        if not np.allclose(noisy_layer, layer_sum):
            noise_added_check = True

    assert noise_added_check, "Noise addition did not change values."
    assert len(noisy_sum) == len(summed_update), "Number of layers changed after noise."
    for i in range(len(noisy_sum)):
        assert noisy_sum[i].shape == summed_update[i].shape, f"Shape mismatch in layer {i} after noise."


try:
    from dp_accounting import dp_event
    from dp_accounting.rdp import RdpAccountant
    ACCOUNTING_LIB_AVAILABLE = True
except ImportError as e:
    import pytest
    pytest.fail(f"DEBUG: dp-accounting import failed: {e}")
    ACCOUNTING_LIB_AVAILABLE = False

import pytest

@pytest.mark.skipif(not ACCOUNTING_LIB_AVAILABLE, reason="dp-accounting library not installed")
def test_privacy_accountant_tracks_budget():
    """Tests privacy budget accumulation using RdpAccountant."""
    num_rounds = 10
    noise_multiplier = 1.5
    sampling_probability = 0.1 # Example: 10 clients selected out of 100
    target_delta = 1e-5

    accountant = RdpAccountant()

    # Define the event for one round (Gaussian mechanism with sampling)
    gauss_event = dp_event.GaussianDpEvent(noise_multiplier)
    sampled_event = dp_event.PoissonSampledDpEvent(sampling_probability, gauss_event)

    epsilons = []
    for _ in range(num_rounds):
        accountant.compose(sampled_event, count=1)
        spent_epsilon = accountant.get_epsilon(target_delta)
        epsilons.append(spent_epsilon)

    # Assertions
    assert len(epsilons) == num_rounds
    assert epsilons[0] > 0 # Epsilon should be positive after first round
    # Epsilon should strictly increase with more compositions
    assert np.all(np.diff(epsilons) > 0)
    print(f"\nPrivacy spend after {num_rounds} rounds (delta={target_delta:.1E}): epsilon={epsilons[-1]:.4f}")
