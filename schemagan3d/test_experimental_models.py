import torch

from .experimental_models import Generator3DUpsample


def test_upsample_generator_shape_and_observation_preservation():
    torch.manual_seed(14)
    sparse = torch.randn(2, 1, 8, 16, 16)
    mask = torch.zeros_like(sparse)
    mask[:, :, :, 3, 5] = 1.0
    sparse = sparse * mask
    output = Generator3DUpsample(base_channels=4)(sparse, mask)
    assert output.shape == sparse.shape
    assert torch.equal(output[mask.bool()], sparse[mask.bool()])
