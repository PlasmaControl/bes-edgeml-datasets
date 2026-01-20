"""Multi-task / multi-objective training with a shared 3D-conv backbone (v6).

This script is intentionally self-contained and defaults to synthetic/random data.

Key features requested
- Shared 3D convolutional encoder (common latent space) for inputs shaped [8, 8, signal_window_size]
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
    main(signal_window_size=256, kernel_t=8, max_epochs=5)

Notes
- This script focuses on the architecture/training mechanics. For real datasets, extend
  Data._load_from_file().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import secrets

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


TaskType = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: TaskType
    output_size: int
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.1

def _validate_task_specs(task_specs: Sequence[TaskSpec]) -> None:
    if not task_specs:
        raise ValueError("task_specs must be non-empty")
    names = [t.name for t in task_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate task names: {names}")
    for spec in task_specs:
        if not isinstance(spec, TaskSpec):
            raise ValueError(f"Each task spec must be a TaskSpec instance, got {type(spec)}")
        if spec.task_type == "multiclass" and spec.output_size < 2:
            raise ValueError(f"multiclass task '{spec.name}' must have output_size >= 2")


@dataclass(kw_only=True, eq=False)
class MultiTask3DDataset(Dataset):
    """Synthetic dataset for multi-task learning.

    Returns
            x: [1, signal_window_size, 8, 8]
      targets: dict[str, Tensor]
        - binary: float targets in {0,1} shaped [output_dim]
        - multiclass: long class index scalar []
        - regression: float targets shaped [output_dim]
    """

    signal_window_size: int
    task_specs: Sequence[TaskSpec]
    seed: int = 123

    x: torch.Tensor = field(init=False, repr=False)
    y: Dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

        if np.log2(self.signal_window_size) % 1 != 0 or self.signal_window_size < 32:
            raise ValueError("signal_window_size must be a power of 2 and >= 32")

        g = torch.Generator().manual_seed(self.seed)

        # x: [N, 1, T, 8, 8]
        self.num_samples = 4096
        self.x = torch.randn(self.num_samples, 1, self.signal_window_size, 8, 8, generator=g)

        # Shared underlying signal (simple): average energy per sample
        shared = self.x.mean(dim=(1, 2, 3, 4))  # [N]

        self.y = {}
        for spec in self.task_specs:
            if spec.task_type == "regression":
                # [N, output_dim]
                w = torch.randn(1, int(spec.output_size), generator=g)
                y = shared.unsqueeze(-1) @ w
                y = y + 0.1 * torch.randn(y.shape, generator=g)
                self.y[spec.name] = y
            elif spec.task_type == "binary":
                logits = shared.unsqueeze(-1).repeat(1, int(spec.output_size))
                logits = logits + 0.5 * torch.randn(logits.shape, generator=g)
                probs = torch.sigmoid(logits)
                self.y[spec.name] = (probs > 0.5).float()
            elif spec.task_type == "multiclass":
                num_classes = int(spec.output_size)
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
    signal_window_size: int = 256
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 123

    def __post_init__(self) -> None:
        super().__init__()

        if np.log2(self.signal_window_size) % 1 != 0 or self.signal_window_size < 32:
            raise ValueError("signal_window_size must be a power of 2 and >= 32")
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
                "signal_window_size": self.signal_window_size,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "seed": self.seed,
            }
        )

        self.train_ds: Dataset = None
        self.val_ds: Dataset = None
        self.test_ds: Dataset = None

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
                signal_window_size=self.signal_window_size,
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
    signal_window_size: int
    backbone_channels: Sequence[int] = (4, 4, 4)
    kernel_t: int|Sequence[int] = 8
    freeze_backbone_epochs: int = 0
    leaky_relu_slope: float = 2e-2
    lr: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__init__()

        if np.log2(self.signal_window_size) % 1 != 0 or self.signal_window_size < 32:
            raise ValueError("signal_window_size must be a power of 2 and >= 32")
        if not self.task_specs:
            raise ValueError("task_specs must be non-empty")
        if self.kernel_t not in (4, 8):
            raise ValueError("kernel_t must be 4 or 8")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs must be >= 0")

        self.save_hyperparameters(
            {
                "signal_window_size": self.signal_window_size,
                "kernel_t": self.kernel_t,
                "backbone_channels": list(self.backbone_channels),
                "freeze_backbone_epochs": self.freeze_backbone_epochs,
                "leaky_relu_slope": self.leaky_relu_slope,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            }
        )

        self.task_names: List[str] = [t.name for t in self.task_specs]
        self.task_specs_by_name: Dict[str, TaskSpec] = {t.name: t for t in self.task_specs}

        # Manual optimization is required for PCGrad.
        self.automatic_optimization = False

        # Shared backbone
        self.backbone = self._build_backbone_encoder(
            kernel_t=self.kernel_t,
            channels=self.backbone_channels,
            leaky_relu_slope=self.leaky_relu_slope,
        )

        # calc latent space size
        self.latent_size = self.backbone(torch.zeros(1, 1, self.signal_window_size, 8, 8)).numel()

        # Task heads
        self.heads = nn.ModuleDict()
        for spec in self.task_specs:
            self.heads[spec.name] = self._build_heads(
                latent_size=self.latent_size,
                output_size=spec.output_size,
                hidden_layer_sizes=spec.hidden_dims,
                dropout=spec.dropout,
                leaky_relu_slope=self.leaky_relu_slope,
            )

        # Homoscedastic uncertainty weighting parameters (log variance per task)
        self.task_log_var = nn.ParameterDict(
            {spec.name: nn.Parameter(torch.zeros(())) for spec in self.task_specs}
        )

        # If requested, start training with a frozen backbone.
        if self.freeze_backbone_epochs > 0:
            self._set_backbone_requires_grad(False)

    @staticmethod
    def _build_backbone_encoder(
        *,
        channels: Sequence[int] = (4, 4, 4),
        kernel_t: int | Sequence[int] = 4,
        leaky_relu_slope: float = 2e-2,
    ) -> nn.Module:
        """Build the shared 3D convolutional encoder.

        Returns a module mapping inputs shaped [B, 1, T, 8, 8] to a flattened feature vector.
        """
        if len(channels) < 1:
            raise ValueError("channels must be non-empty")

        if isinstance(kernel_t, Sequence):
            kernel_ts = list(kernel_t)
            if len(kernel_ts) != len(channels):
                raise ValueError("If kernel_t is a sequence, it must have the same length as channels")
        elif isinstance(kernel_t, int):
            kernel_ts = [kernel_t] * len(channels)
        else:
            raise ValueError("kernel_t must be an int or a sequence of ints")

        for k in kernel_ts:
            if k not in (4, 8):
                raise ValueError("Each kernel_t must be 4 or 8")

        blocks: List[nn.Module] = []
        c_in = 1
        for c_out, kT in zip(channels, kernel_ts):
            blocks.append(
                nn.Conv3d(
                    c_in,
                    c_out,
                    kernel_size=(kT, 3, 3),
                    stride=(kT, 1, 1),
                    padding="valid",
                    bias=True,
                )
            )
            blocks.append(nn.GroupNorm(num_groups=1, num_channels=c_out))
            blocks.append(nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=False))
            c_in = c_out

        blocks.append(nn.Flatten(1))
        return nn.Sequential(*blocks)

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    def on_train_epoch_start(self) -> None:
        # Freeze for the first N epochs (0-indexed). Unfreeze afterwards.
        if self.freeze_backbone_epochs <= 0:
            return

        if self.current_epoch < self.freeze_backbone_epochs:
            self._set_backbone_requires_grad(False)
        else:
            self._set_backbone_requires_grad(True)

    @staticmethod
    def _build_heads(
        *,
        latent_size: int,
        output_size: int,
        hidden_layer_sizes: Sequence[int],
        dropout: float,
        leaky_relu_slope: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = latent_size
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=False))
            if dropout and dropout > 0:
                layers.append(nn.Dropout1d(p=dropout, inplace=False))
            prev = h

        layers.append(nn.Linear(prev, output_size))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, Latent_Size]
        latent_z = self.backbone(x)
        return {name: head(latent_z) for name, head in self.heads.items()}

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
            # retain = i != (len(tasks) - 1)
            grads = torch.autograd.grad(
                task_objectives[t],
                shared_params,
                retain_graph=True,
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

    def eval_step(self, substage: str, batch: Any, batch_idx: int) -> Tuple[torch.Tensor]:
        x: torch.Tensor = batch[0]  # should be [B, 1, signal_window_size, 8, 8]
        targets: torch.Tensor = batch[1]

        outputs = self(x)
        task_losses = self._compute_task_losses(outputs, targets)
        total, weighted = self._uncertainty_weighted_losses(task_losses)

        self.log(f"loss/{substage}", total.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for name in self.task_names:
            if name in task_losses:
                self.log(f"loss_raw/{substage}/{name}", task_losses[name].detach(), on_step=False, on_epoch=True, sync_dist=True)
            if name in weighted:
                self.log(
                    f"loss_weighted/{substage}/{name}",
                    weighted[name].detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self._log_task_logvars(prefix=f"{substage}/")

        return total.detach(), weighted

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        opt: torch.optim.Optimizer = self.optimizers()
        opt.zero_grad(set_to_none=True)

        total, weighted = self.eval_step("train", batch, batch_idx)

        shared_params = self._get_shared_params()
        self._pcgrad_step(task_objectives=weighted, shared_params=shared_params)
        opt.step()

        return total

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        _ = self.eval_step("val", batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def main(
    *,
    data_path: Optional[str] = None,
    signal_window_size: int = 4*4*4*2,
    kernel_t: int|Sequence[int] = 4,
    backbone_channels: Sequence[int] = (4, 4, 4),
    freeze_backbone_epochs: int = 0,
    task_specs: Optional[Sequence[TaskSpec]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: Optional[int] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 2,
    accelerator: str = "auto",
    devices: str = "auto",
    log_dir: str = "./lightning_logs",
) -> None:
    """Entry point (programmatic only).

    This module does not implement a command-line interface; configure runs by
    passing keyword arguments into main().
    """

    if seed is None:
        # Choose a random 31-bit seed (compatible with common libs expecting a signed int).
        seed = secrets.randbits(31)

    pl.seed_everything(seed, workers=True)

    if np.log2(signal_window_size) % 1 != 0 or signal_window_size < 32:
        raise ValueError("signal_window_size must be a power of 2 and >= 32")

    if not task_specs:
        task_specs = [
            TaskSpec(name="binary", task_type="binary", output_size=1, hidden_dims=(128, 64), dropout=0.1),
            TaskSpec(name="multiclass", task_type="multiclass", output_size=4, hidden_dims=(128, 64), dropout=0.1),
            TaskSpec(name="regression", task_type="regression", output_size=1, hidden_dims=(128, 64), dropout=0.1),
        ]
    _validate_task_specs(task_specs)

    data = Data(
        task_specs=task_specs,
        signal_window_size=signal_window_size,
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = Model(
        task_specs=task_specs,
        signal_window_size=signal_window_size,
        kernel_t=kernel_t,
        backbone_channels=backbone_channels,
        freeze_backbone_epochs=freeze_backbone_epochs,
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
