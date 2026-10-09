from __future__ import annotations

import torch

from WIFIJEPA.src.masking import make_link_mask
from WIFIJEPA.src.model import StructuredPoseModel, WiFiJEPA


def test_link_mask_covers_whole_antenna():
    mask = make_link_mask(8, n_time=10, n_links=3, n_masked_links=1)
    mask = mask.view(8, 10, 3)
    assert mask.sum(dim=2).eq(1).all()
    assert mask.sum(dim=1).eq(mask[:, :1].sum(dim=1) * 10).all()


def test_jepa_forward_and_target_has_no_gradient():
    model = WiFiJEPA(embed_dim=48, depth=2, num_heads=4, ffn_dim=96, predictor_depth=1)
    x = torch.randn(4, 3, 114, 10)
    loss, log = model(x)
    loss.backward()
    assert loss.ndim == 0
    assert log["masked_tokens"] == 40
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    model.update_target()


def test_structured_pose_model_shape():
    model = StructuredPoseModel(
        dropout_p=0.0, embed_dim=48, depth=2, num_heads=4, ffn_dim=96
    )
    output = model(torch.randn(2, 3, 114, 10))
    output_5d = model(torch.randn(2, 1, 3, 114, 10))
    assert output.shape == (2, 17, 3)
    assert output_5d.shape == (2, 17, 3)


if __name__ == "__main__":
    test_link_mask_covers_whole_antenna()
    test_jepa_forward_and_target_has_no_gradient()
    test_structured_pose_model_shape()
    print("WIFIJEPA core smoke passed")
