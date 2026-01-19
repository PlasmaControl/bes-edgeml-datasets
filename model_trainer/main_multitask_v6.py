"""Multi-task / multi-objective training with a shared 3D-conv backbone (v6).

This script is intentionally self-contained and defaults to synthetic/random data.

Key features requested
- Shared 3D convolutional encoder (common latent space) for inputs shaped [8, 8, n_time]
  (implemented as torch tensors [B, 1, n_time, 8, 8] for Conv3d).
- Conv3d kernel sizes (time, height, width) = (kT, 3, 3) with kT in {4, 8}.
  Stride is (kT, 1, 1) (stride equals the kernel time dimension).
- GroupNorm after conv layers (configured as GroupNorm(1, C) = across all channels).
- LayerNorm on the flattened feature vector (latent vector).
- Task-specific MLP heads for: binary classification, multiclass classification, regression.
  In heads: Linear -> BatchNorm1d -> LeakyReLU -> Dropout1d for hidden layers.
- Homoscedastic Uncertainty Weighting (learnable per-task log-variance).
- Gradient conflict resolution via PCGrad on shared backbone parameters.
- LightningModule `Model` + LightningDataModule `Data` + runnable main().

Example
    # Programmatic usage (keyword defaults)
    from model_trainer.main_multitask_v6 import main
    main(n_time=256, kernel_t=8, max_epochs=5)

Notes
- This script focuses on the architecture/training mechanics. For real datasets, extend
  Data._load_from_file().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from torchmetrics import MeanMetric

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


TaskType = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: TaskType
    output_dim: int
    head_hidden_dims: Tuple[int, ...] = (128, 64)
    head_dropout: float = 0.1


def _parse_task_specs(tasks_json: Optional[str]) -> List[TaskSpec]:
    if not tasks_json:
        return [
            TaskSpec(name="binary", task_type="binary", output_dim=1, head_hidden_dims=(128, 64), head_dropout=0.1),
            TaskSpec(name="multiclass", task_type="multiclass", output_dim=4, head_hidden_dims=(128, 64), head_dropout=0.1),
            TaskSpec(name="regression", task_type="regression", output_dim=1, head_hidden_dims=(128, 64), head_dropout=0.1),
        ]

    # Accept either a JSON string or a path to a JSON file.
    raw: Any
    if tasks_json.strip().endswith(".json"):
        with open(tasks_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(tasks_json)

    if not isinstance(raw, list):
        raise ValueError("tasks_json must decode to a list of task spec dicts")

    specs: List[TaskSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each task spec must be a dict")
        specs.append(
            TaskSpec(
                name=str(item["name"]),
                task_type=str(item["task_type"]),
                output_dim=int(item["output_dim"]),
                head_hidden_dims=tuple(int(x) for x in item.get("head_hidden_dims", (128, 64))),
                head_dropout=float(item.get("head_dropout", 0.1)),
            )
        )

    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate task names: {names}")

    return specs


@dataclass(kw_only=True, eq=False)
class Conv3DBackbone(nn.Module):
    """Shared 3D convolutional encoder producing a latent feature vector."""

    in_channels: int = 1
    kernel_t: int = 8
    channels: Sequence[int] = (16, 32, 64)
    latent_dim: int = 128
    leaky_relu_slope: float = 2e-2

    conv: nn.Sequential = field(init=False, repr=False)
    pool: nn.AdaptiveAvgPool3d = field(init=False, repr=False)
    flat_norm: nn.LayerNorm = field(init=False, repr=False)
    to_latent: nn.Sequential = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

        if self.kernel_t not in (4, 8):
            raise ValueError("kernel_t must be 4 or 8")
        if len(self.channels) < 1:
            raise ValueError("channels must be non-empty")

        blocks: List[nn.Module] = []
        c_in = self.in_channels
        for c_out in self.channels:
            blocks.append(
                nn.Conv3d(
                    c_in,
                    c_out,
                    kernel_size=(self.kernel_t, 3, 3),
                    stride=(self.kernel_t, 1, 1),
                    padding=(0, 1, 1),
                    bias=True,
                )
            )
            blocks.append(nn.GroupNorm(num_groups=1, num_channels=c_out))
            blocks.append(nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True))
            c_in = c_out

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Flattened feature vector normalization
        self.flat_norm = nn.LayerNorm(self.channels[-1])

        # Project into a user-controlled latent space
        self.to_latent = nn.Sequential(
            nn.Linear(self.channels[-1], self.latent_dim),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),
            nn.LayerNorm(self.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, n_time, 8, 8]
        if x.ndim != 5:
            raise ValueError(f"Expected input rank 5 [B,1,T,8,8], got {tuple(x.shape)}")
        y = self.conv(x)
        y = self.pool(y)
        y = y.flatten(1)  # [B, C]
        y = self.flat_norm(y)
        z = self.to_latent(y)
        return z


@dataclass(kw_only=True, eq=False)
class MultiTask3DDataset(Dataset):
    """Synthetic dataset for multi-task learning.

    Returns
      x: [1, n_time, 8, 8]
      targets: dict[str, Tensor]
        - binary: float targets in {0,1} shaped [output_dim]
        - multiclass: long class index scalar []
        - regression: float targets shaped [output_dim]
    """

    num_samples: int
    n_time: int
    task_specs: Sequence[TaskSpec]
    seed: int = 123

    x: torch.Tensor = field(init=False, repr=False)
    y: Dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

        if self.n_time < 128:
            raise ValueError("n_time must be >= 128")

        g = torch.Generator().manual_seed(self.seed)

        # x: [N, 1, T, 8, 8]
        self.x = torch.randn(self.num_samples, 1, self.n_time, 8, 8, generator=g)

        # Shared underlying signal (simple): average energy per sample
        shared = self.x.mean(dim=(1, 2, 3, 4))  # [N]

        self.y = {}
        for spec in self.task_specs:
            if spec.task_type == "regression":
                # [N, output_dim]
                w = torch.randn(1, int(spec.output_dim), generator=g)
                y = shared.unsqueeze(-1) @ w
                y = y + 0.1 * torch.randn(y.shape, generator=g)
                self.y[spec.name] = y
            elif spec.task_type == "binary":
                logits = shared.unsqueeze(-1).repeat(1, int(spec.output_dim))
                logits = logits + 0.5 * torch.randn(logits.shape, generator=g)
                probs = torch.sigmoid(logits)
                self.y[spec.name] = (probs > 0.5).float()
            elif spec.task_type == "multiclass":
                num_classes = int(spec.output_dim)
                if num_classes < 2:
                    raise ValueError(f"multiclass task '{spec.name}' must have output_dim >= 2")
                class_logits = shared.unsqueeze(-1) + 0.2 * torch.arange(num_classes).unsqueeze(0)
                class_logits = class_logits + 0.5 * torch.randn(class_logits.shape, generator=g)
                idx = torch.argmax(class_logits, dim=-1)
                self.y[spec.name] = idx.long()
            else:
                raise ValueError(f"Unknown task type: {spec.task_type}")

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.x[idx]
        targets = {k: v[idx] for k, v in self.y.items()}
        return x, targets


@dataclass(kw_only=True, eq=False)
class Data(pl.LightningDataModule):
    task_specs: Sequence[TaskSpec]
    data_path: Optional[str] = None
    n_time: int = 256
    num_samples: int = 4096
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 123

    train_ds: Optional[Dataset] = field(default=None, init=False, repr=False)
    val_ds: Optional[Dataset] = field(default=None, init=False, repr=False)
    test_ds: Optional[Dataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

        if self.n_time < 128:
            raise ValueError("n_time must be >= 128")
        if not self.task_specs:
            raise ValueError("task_specs must be non-empty")
        names = [t.name for t in self.task_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate task names: {names}")

        # save_hyperparameters relies on the caller frame locals; __post_init__ doesn't
        # have the init args as locals, so we pass an explicit payload.
        self.save_hyperparameters(
            {
                "data_path": self.data_path,
                "n_time": self.n_time,
                "num_samples": self.num_samples,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "seed": self.seed,
            }
        )

    def _load_from_file(self) -> Dataset:
        """Load a dataset from disk.

        This is left intentionally minimal; extend to match your data formats.
        Currently supports a torch file containing {'x': Tensor, 'y': dict[str, Tensor]}.
        """
        if self.data_path is None:
            raise RuntimeError("data_path is None")

        path = str(self.data_path)
        if not path.endswith(".pt"):
            raise NotImplementedError(
                "Only .pt files are supported by default. "
                "Provide no --data_path to use random data, or extend _load_from_file()."
            )

        blob = torch.load(path, map_location="cpu")
        x = blob.get("x", None)
        y = blob.get("y", None)
        if x is None or y is None:
            raise ValueError("Expected torch file with keys 'x' and 'y'")
        if x.shape[-2:] != (8, 8):
            raise ValueError(f"Expected spatial shape 8x8, got {tuple(x.shape)}")

        class _Loaded(Dataset):
            def __init__(self, x_: torch.Tensor, y_: Dict[str, torch.Tensor]):
                self.x = x_
                self.y = y_

            def __len__(self) -> int:
                return int(self.x.shape[0])

            def __getitem__(self, idx: int):
                return self.x[idx], {k: v[idx] for k, v in self.y.items()}

        return _Loaded(x, y)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_path:
            full: Dataset = self._load_from_file()
        else:
            full = MultiTask3DDataset(
                num_samples=self.num_samples,
                n_time=self.n_time,
                task_specs=self.task_specs,
                seed=self.seed,
            )

        n = len(full)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


@dataclass(kw_only=True, eq=False)
class Model(pl.LightningModule):
    task_specs: Sequence[TaskSpec]
    n_time: int
    kernel_t: int = 8
    backbone_channels: Sequence[int] = (16, 32, 64)
    latent_dim: int = 128
    leaky_relu_slope: float = 2e-2
    lr: float = 1e-3
    weight_decay: float = 1e-4

    task_specs_by_name: Dict[str, TaskSpec] = field(default_factory=dict, init=False, repr=False)
    task_names: List[str] = field(default_factory=list, init=False, repr=False)

    backbone: Conv3DBackbone = field(init=False, repr=False)
    heads: nn.ModuleDict = field(init=False, repr=False)
    task_log_var: nn.ParameterDict = field(init=False, repr=False)

    train_loss_total: MeanMetric = field(init=False, repr=False)
    val_loss_total: MeanMetric = field(init=False, repr=False)
    train_loss_raw: nn.ModuleDict = field(init=False, repr=False)
    val_loss_raw: nn.ModuleDict = field(init=False, repr=False)
    train_loss_weighted: nn.ModuleDict = field(init=False, repr=False)
    val_loss_weighted: nn.ModuleDict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

        if self.n_time < 128:
            raise ValueError("n_time must be >= 128")
        if self.kernel_t not in (4, 8):
            raise ValueError("kernel_t must be 4 or 8")
        if not self.task_specs:
            raise ValueError("task_specs must be non-empty")

        names = [t.name for t in self.task_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate task names: {names}")
        self.task_names = names
        self.task_specs_by_name = {t.name: t for t in self.task_specs}

        self.save_hyperparameters(
            {
                "n_time": self.n_time,
                "kernel_t": self.kernel_t,
                "backbone_channels": list(self.backbone_channels),
                "latent_dim": self.latent_dim,
                "leaky_relu_slope": self.leaky_relu_slope,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }
        )

        # Manual optimization is required for PCGrad.
        self.automatic_optimization = False

        # Shared backbone
        self.backbone = Conv3DBackbone(
            in_channels=1,
            kernel_t=self.kernel_t,
            channels=self.backbone_channels,
            latent_dim=self.latent_dim,
            leaky_relu_slope=self.leaky_relu_slope,
        )

        # Task heads
        self.heads = nn.ModuleDict()
        for spec in self.task_specs:
            self.heads[spec.name] = self._build_head(
                input_dim=self.latent_dim,
                output_dim=spec.output_dim,
                hidden_dims=spec.head_hidden_dims,
                dropout=spec.head_dropout,
                leaky_relu_slope=self.leaky_relu_slope,
            )

        # Homoscedastic uncertainty weighting parameters (log variance per task)
        self.task_log_var = nn.ParameterDict(
            {spec.name: nn.Parameter(torch.zeros(())) for spec in self.task_specs}
        )

        # --- TorchMetrics loss tracking ---
        metric_kwargs = {
            "dist_sync_on_step": False,
            "sync_on_compute": True,
        }

        self.train_loss_total = MeanMetric(**metric_kwargs)
        self.val_loss_total = MeanMetric(**metric_kwargs)

        self.train_loss_raw = nn.ModuleDict({name: MeanMetric(**metric_kwargs) for name in self.task_names})
        self.val_loss_raw = nn.ModuleDict({name: MeanMetric(**metric_kwargs) for name in self.task_names})

        self.train_loss_weighted = nn.ModuleDict({name: MeanMetric(**metric_kwargs) for name in self.task_names})
        self.val_loss_weighted = nn.ModuleDict({name: MeanMetric(**metric_kwargs) for name in self.task_names})

    @staticmethod
    def _build_head(
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        leaky_relu_slope: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout1d(p=dropout, inplace=True))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, 1, T, 8, 8]
        z = self.backbone(x)
        return {name: head(z) for name, head in self.heads.items()}

    def _compute_task_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        for spec in self.task_specs:
            pred = outputs[spec.name]
            tgt = targets[spec.name]

            if spec.task_type == "regression":
                # pred/tgt: [B, output_dim]
                losses[spec.name] = F.mse_loss(pred, tgt)
            elif spec.task_type == "binary":
                # pred/tgt: [B, output_dim]
                losses[spec.name] = F.binary_cross_entropy_with_logits(pred, tgt)
            elif spec.task_type == "multiclass":
                # pred: [B, C]; tgt: [B] (class index) OR [B,C] (one-hot)
                if tgt.ndim == 2:
                    tgt_idx = torch.argmax(tgt, dim=-1)
                else:
                    tgt_idx = tgt
                losses[spec.name] = F.cross_entropy(pred, tgt_idx.long())
            else:
                raise ValueError(f"Unknown task type: {spec.task_type}")

        return losses

    def _uncertainty_weighted_losses(
        self, task_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Kendall et al. homoscedastic uncertainty weighting.

        For each task i:
          L_i' = exp(-s_i) * L_i + s_i, where s_i = log(sigma_i^2)
        """
        total = torch.zeros((), device=self.device)
        weighted: Dict[str, torch.Tensor] = {}
        for name, loss in task_losses.items():
            log_var = self.task_log_var[name]
            precision = torch.exp(-log_var)
            wloss = precision * loss + log_var
            weighted[name] = wloss
            total = total + wloss
        return total, weighted

    def _get_shared_params(self) -> List[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    @torch.no_grad()
    def _log_task_logvars(self, prefix: str) -> None:
        for name in self.task_names:
            self.log(f"{prefix}log_var/{name}", self.task_log_var[name].detach(), on_step=False, on_epoch=True)

    def _pcgrad_step(self, task_objectives: Dict[str, torch.Tensor], shared_params: List[nn.Parameter]) -> None:
        """PCGrad on shared parameters.

        task_objectives should be per-task scalar objectives (already uncertainty-weighted here).
        """
        tasks = [t for t in self.task_names if t in task_objectives]
        if len(tasks) <= 1 or not shared_params:
            self.manual_backward(sum(task_objectives.values()))
            return

        sizes = [int(p.numel()) for p in shared_params]
        dtypes = [p.dtype for p in shared_params]

        grads_by_task_flat: Dict[str, torch.Tensor] = {}
        for i, t in enumerate(tasks):
            retain = i != (len(tasks) - 1)
            grads = torch.autograd.grad(
                task_objectives[t],
                shared_params,
                retain_graph=retain,
                allow_unused=True,
            )
            flat_parts: List[torch.Tensor] = []
            for j, (g, n) in enumerate(zip(grads, sizes)):
                if g is None:
                    flat_parts.append(torch.zeros((n,), device=self.device, dtype=dtypes[j]))
                else:
                    flat_parts.append(g.reshape(-1))
            grads_by_task_flat[t] = torch.cat(flat_parts, dim=0)

        eps = 1e-12
        for ti in tasks:
            gi = grads_by_task_flat[ti]
            for tj in tasks:
                if ti == tj:
                    continue
                gj = grads_by_task_flat[tj]
                dot = torch.dot(gi, gj)
                if dot < 0:
                    denom = torch.dot(gj, gj).clamp_min(eps)
                    gi = gi - (dot / denom) * gj
            grads_by_task_flat[ti] = gi

        merged_flat = sum((grads_by_task_flat[t] for t in tasks))

        merged_grads: List[torch.Tensor] = []
        offset = 0
        for p, n in zip(shared_params, sizes):
            merged_grads.append(merged_flat[offset : offset + n].reshape_as(p))
            offset += n

        # Backprop the full combined objective to populate head grads,
        # then override the shared grads with PCGrad merged grads.
        total_obj = sum(task_objectives.values())
        self.manual_backward(total_obj)
        for p, g in zip(shared_params, merged_grads):
            p.grad = g.detach()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        x, targets = batch
        # Dataset returns x: [1,T,8,8]; make it [B,1,T,8,8]
        if x.ndim == 4:
            x = x.unsqueeze(1)

        outputs = self(x)
        task_losses = self._compute_task_losses(outputs, targets)
        total, weighted = self._uncertainty_weighted_losses(task_losses)

        # Update TorchMetrics loss trackers (detach: metrics are for logging only).
        self.train_loss_total(total.detach())
        for name, l in task_losses.items():
            if name in self.train_loss_raw:
                self.train_loss_raw[name](l.detach())
        for name, l in weighted.items():
            if name in self.train_loss_weighted:
                self.train_loss_weighted[name](l.detach())

        shared_params = self._get_shared_params()
        self._pcgrad_step(task_objectives=weighted, shared_params=shared_params)

        opt.step()

        # Log via TorchMetrics (epoch-wise aggregation).
        self.log("loss/train", self.train_loss_total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for name in self.task_names:
            self.log(f"loss_raw/train/{name}", self.train_loss_raw[name], on_step=False, on_epoch=True, sync_dist=True)
            self.log(
                f"loss_weighted/train/{name}",
                self.train_loss_weighted[name],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self._log_task_logvars(prefix="train/")

        return total.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, targets = batch
        if x.ndim == 4:
            x = x.unsqueeze(1)

        outputs = self(x)
        task_losses = self._compute_task_losses(outputs, targets)
        total, weighted = self._uncertainty_weighted_losses(task_losses)

        # Update + log via TorchMetrics.
        self.val_loss_total(total.detach())
        for name, l in task_losses.items():
            if name in self.val_loss_raw:
                self.val_loss_raw[name](l.detach())
        for name, l in weighted.items():
            if name in self.val_loss_weighted:
                self.val_loss_weighted[name](l.detach())

        self.log("loss/val", self.val_loss_total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for name in self.task_names:
            self.log(f"loss_raw/val/{name}", self.val_loss_raw[name], on_step=False, on_epoch=True, sync_dist=True)
            self.log(
                f"loss_weighted/val/{name}",
                self.val_loss_weighted[name],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self._log_task_logvars(prefix="val/")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def main(
    *,
    data_path: Optional[str] = None,
    n_time: int = 256,
    kernel_t: int = 8,
    backbone_channels: Sequence[int] = (16, 32, 64),
    latent_dim: int = 128,
    tasks_json: Optional[str] = None,
    batch_size: int = 32,
    num_samples: int = 4096,
    num_workers: int = 0,
    seed: int = 123,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 10,
    accelerator: str = "auto",
    devices: str = "auto",
    log_dir: str = "./lightning_logs",
) -> None:
    """Entry point (programmatic only).

    This module does not implement a command-line interface; configure runs by
    passing keyword arguments into main().
    """

    if n_time < 128:
        raise ValueError("n_time must be >= 128")

    task_specs = _parse_task_specs(tasks_json)

    pl.seed_everything(seed, workers=True)

    data = Data(
        task_specs=task_specs,
        n_time=n_time,
        data_path=data_path,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = Model(
        task_specs=task_specs,
        n_time=n_time,
        kernel_t=kernel_t,
        backbone_channels=backbone_channels,
        latent_dim=latent_dim,
        lr=lr,
        weight_decay=weight_decay,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="multitask_v6")

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(monitor="loss/val", mode="min", save_top_k=1),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    main()
