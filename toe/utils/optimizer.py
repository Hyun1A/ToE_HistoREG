import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------
def build_optimizer(
    model: torch.nn.Module,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-2,
    betas: tuple = (0.9, 0.999),
) -> AdamW:

    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen
        # heuristic: backbone 관련 이름 패턴
        if any(
            kw in name
            for kw in (
                "backbone",
                "encoder",
                "conv",
                "patch_embed",
                "stem",
            )
        ):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]
    return AdamW(param_groups, betas=betas)


# ---------------------------------------------------------------------
# Scheduler (optional)
# ---------------------------------------------------------------------


class CosineAnnealingLRPerGroup(_LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_mins, last_epoch: int = -1):
        """
        eta_mins: float 또는 list[float] (param_groups와 동일 길이)
        """
        if isinstance(eta_mins, (int, float)):
            eta_mins = [float(eta_mins)] * len(optimizer.param_groups)
        else:
            assert len(eta_mins) == len(optimizer.param_groups), \
                "eta_mins length must match param_groups"

        self.T_max = T_max
        self.eta_mins = list(map(float, eta_mins))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.T_max)
        factor = (1 + math.cos(math.pi * t / self.T_max)) / 2.0
        lrs = []
        for base_lr, eta_min in zip(self.base_lrs, self.eta_mins):
            lrs.append(eta_min + (base_lr - eta_min) * factor)
        return lrs


def build_scheduler(optimizer, max_epochs: int, min_lr_ratio: float = 0.01):
    eta_mins = [g["lr"] * min_lr_ratio for g in optimizer.param_groups]
    return CosineAnnealingLRPerGroup(optimizer, T_max=max_epochs, eta_mins=eta_mins)