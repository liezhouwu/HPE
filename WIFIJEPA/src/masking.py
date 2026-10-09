from __future__ import annotations

import torch


def make_link_mask(
    batch_size: int,
    n_time: int = 10,
    n_links: int = 3,
    n_masked_links: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """为每个样本随机选择整条天线链路，返回 (B, T*L) 的目标掩码。"""
    links = torch.rand(batch_size, n_links, device=device).argsort(dim=1)
    links = links[:, :n_masked_links]
    link_mask = torch.zeros(batch_size, n_links, dtype=torch.bool, device=device)
    link_mask.scatter_(1, links, True)
    return link_mask[:, None, :].expand(batch_size, n_time, n_links).reshape(batch_size, -1)
