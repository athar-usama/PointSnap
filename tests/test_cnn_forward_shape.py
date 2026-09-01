import torch

from pointsnap.detect.cnn import ConfidenceUNet
from pointsnap.detect.features import NUM_CHANNELS


def test_forward_shape_matches_input_spatial_size():
    model = ConfidenceUNet()
    x = torch.randn(2, NUM_CHANNELS, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_forward_works_on_non_square_multiple_of_eight():
    model = ConfidenceUNet()
    x = torch.randn(1, NUM_CHANNELS, 32, 48)
    out = model(x)
    assert out.shape == (1, 1, 32, 48)


def test_parameter_count_is_within_the_planned_small_budget():
    model = ConfidenceUNet()
    n = model.num_parameters()
    assert 200_000 < n < 1_500_000
