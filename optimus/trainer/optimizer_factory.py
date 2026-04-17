from __future__ import annotations

import torch
from torch.optim import Optimizer

from optimus.trainer.configuration.train import TrainConfig


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> Optimizer:
    """Build an optimizer from the training configuration.

    Supported values for ``config.optimizer``:
    - ``"AdamW"`` (default) — :class:`torch.optim.AdamW`.
      Uses ``lr``, ``weight_decay``, ``beta1``, ``beta2``, ``eps``, ``fused``.
    - ``"Adam"`` — :class:`torch.optim.Adam`.
      Uses ``lr``, ``weight_decay``, ``beta1``, ``beta2``, ``eps``, ``fused``.
    - ``"SGD"`` — :class:`torch.optim.SGD`.
      Uses ``lr``, ``weight_decay``; momentum is fixed at ``0.9``.

    Args:
        model: The model whose parameters will be optimised.
        config: Training configuration dataclass.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If ``config.optimizer`` is not a recognised name.
    """
    params = model.parameters()
    name = config.optimizer

    if name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            fused=config.fused,
        )
    elif name == "Adam":
        return torch.optim.Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            fused=config.fused,
        )
    elif name == "SGD":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}. "
            "Supported values: 'AdamW', 'Adam', 'SGD'."
        )
