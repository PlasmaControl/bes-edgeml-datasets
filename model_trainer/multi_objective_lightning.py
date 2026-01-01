"""Multi-objective (multi-task) training with a shared latent space using PyTorch Lightning.

Key features
- Arbitrary number of tasks.
- Shared encoder (MLP trunk) that produces a latent feature space.
- Per-task heads (MLPs) supporting:
  - regression
  - binary classification
  - multi-class classification (one-hot targets with arbitrary number of classes)
- LightningModule + LightningDataModule + Dataset + runnable main().

This file is intentionally self-contained and uses a synthetic dataset by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

try:
    # Optional dependency
    from lightning.pytorch.loggers import WandbLogger  # type: ignore
except Exception:  # pragma: no cover
    WandbLogger = None  # type: ignore


TaskType = Literal["regression", "binary", "multiclass"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: TaskType
    output_dim: int
    weight: float = 1.0
    head_hidden_dims: Tuple[int, ...] = (64, 64)


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class MultiTaskDataset(Dataset):
    """Simple dataset that returns (x, targets_dict).

    Default behavior: generates a synthetic dataset for the provided task specs.
    Optionally, pass explicit tensors to use real data.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        input_dim: int,
        task_specs: Sequence[TaskSpec],
        x: Optional[torch.Tensor] = None,
        y: Optional[Dict[str, torch.Tensor]] = None,
        seed: int = 123,
    ) -> None:
        super().__init__()
        self.task_specs = list(task_specs)

        if x is not None and y is not None:
            if x.ndim != 2:
                raise ValueError("x must be 2D: [N, input_dim]")
            if x.shape[0] <= 0:
                raise ValueError("x must have at least 1 sample")
            self.x = x
            self.y = y
            return

        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, input_dim, generator=g)

        # Build synthetic targets with a shared underlying linear signal.
        w_shared = torch.randn(input_dim, 1, generator=g)
        shared_signal = (self.x @ w_shared).squeeze(-1)  # [N]

        self.y = {}
        for spec in self.task_specs:
            if spec.task_type == "regression":
                # Regression targets: [N, output_dim]
                w = torch.randn(1, spec.output_dim, generator=g)
                y_reg = shared_signal.unsqueeze(-1) @ w
                y_reg = y_reg + 0.1 * torch.randn(
                    y_reg.shape,
                    generator=g,
                    device=y_reg.device,
                    dtype=y_reg.dtype,
                )
                self.y[spec.name] = y_reg
            elif spec.task_type == "binary":
                # Binary targets: [N, output_dim] in {0,1}
                logits = shared_signal.unsqueeze(-1).repeat(1, spec.output_dim)
                logits = logits + 0.5 * torch.randn(
                    logits.shape,
                    generator=g,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                probs = torch.sigmoid(logits)
                self.y[spec.name] = (probs > 0.5).float()
            elif spec.task_type == "multiclass":
                # One-hot targets: [N, num_classes]
                num_classes = spec.output_dim
                if num_classes < 2:
                    raise ValueError(f"multiclass task '{spec.name}' must have output_dim >= 2")
                # Create class logits as simple affine projections of shared signal
                class_logits = torch.stack(
                    [
                        shared_signal + 0.2 * i
                        for i in range(num_classes)
                    ],
                    dim=-1,
                )
                class_logits = class_logits + 0.5 * torch.randn(
                    class_logits.shape,
                    generator=g,
                    device=class_logits.device,
                    dtype=class_logits.dtype,
                )
                class_idx = torch.argmax(class_logits, dim=-1)
                self.y[spec.name] = F.one_hot(class_idx, num_classes=num_classes).float()
            else:
                raise ValueError(f"Unknown task type: {spec.task_type}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.x[idx], {k: v[idx] for k, v in self.y.items()}


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        num_samples: int,
        input_dim: int,
        task_specs: Sequence[TaskSpec],
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 123,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.task_specs = list(task_specs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        full = MultiTaskDataset(
            num_samples=self.num_samples,
            input_dim=self.input_dim,
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

    def predict_dataloader(self) -> DataLoader:
        # Predict on test split by default
        return self.test_dataloader()


class MultiObjectiveModel(pl.LightningModule):
    """LightningModule for multi-objective learning with a shared latent space."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        task_specs: Sequence[TaskSpec],
        trunk_hidden_dims: Sequence[int] = (256, 256),
        trunk_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["task_specs"])

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.task_specs = list(task_specs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Shared encoder / trunk / backbone: MLP -> latent
        self.trunk = _build_mlp(
            input_dim=input_dim,
            hidden_dims=trunk_hidden_dims,
            output_dim=latent_dim,
            dropout=trunk_dropout,
        )

        # Task heads: per-task MLP from latent -> output
        self.heads = nn.ModuleDict()
        for spec in self.task_specs:
            self.heads[spec.name] = self._build_task_head(spec)

    # --- Head builders (explicit methods requested) ---

    def make_regression_head(self, *, latent_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
        return _build_mlp(latent_dim, hidden_dims, output_dim, dropout=0.0)

    def make_binary_classification_head(
        self, *, latent_dim: int, output_dim: int, hidden_dims: Sequence[int]
    ) -> nn.Sequential:
        # Outputs logits; loss uses BCEWithLogits
        return _build_mlp(latent_dim, hidden_dims, output_dim, dropout=0.0)

    def make_multiclass_classification_head(
        self, *, latent_dim: int, num_classes: int, hidden_dims: Sequence[int]
    ) -> nn.Sequential:
        # Outputs logits of shape [B, C]; targets are one-hot [B, C]
        return _build_mlp(latent_dim, hidden_dims, num_classes, dropout=0.0)

    def _build_task_head(self, spec: TaskSpec) -> nn.Sequential:
        if spec.task_type == "regression":
            return self.make_regression_head(
                latent_dim=self.latent_dim,
                output_dim=spec.output_dim,
                hidden_dims=spec.head_hidden_dims,
            )
        if spec.task_type == "binary":
            return self.make_binary_classification_head(
                latent_dim=self.latent_dim,
                output_dim=spec.output_dim,
                hidden_dims=spec.head_hidden_dims,
            )
        if spec.task_type == "multiclass":
            return self.make_multiclass_classification_head(
                latent_dim=self.latent_dim,
                num_classes=spec.output_dim,
                hidden_dims=spec.head_hidden_dims,
            )
        raise ValueError(f"Unknown task type: {spec.task_type}")

    # --- Forward ---

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        return {name: head(z) for name, head in self.heads.items()}

    # --- Losses ---

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    def binary_classification_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        return F.binary_cross_entropy_with_logits(logits, target)

    def multiclass_onehot_loss(self, logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        """Cross-entropy for one-hot targets.

        logits: [B, C]
        target_onehot: [B, C] (float), where rows sum to 1.
        """
        if logits.ndim != 2 or target_onehot.ndim != 2:
            raise ValueError("multiclass expects 2D tensors: logits [B,C], target_onehot [B,C]")
        if logits.shape != target_onehot.shape:
            raise ValueError(f"Shape mismatch: logits {tuple(logits.shape)} vs target {tuple(target_onehot.shape)}")

        log_probs = F.log_softmax(logits, dim=-1)
        per_example = -(target_onehot * log_probs).sum(dim=-1)
        return per_example.mean()

    def compute_total_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = torch.zeros((), device=self.device)
        per_task: Dict[str, torch.Tensor] = {}

        for spec in self.task_specs:
            pred = outputs[spec.name]
            target = targets[spec.name]

            if spec.task_type == "regression":
                loss = self.regression_loss(pred, target)
            elif spec.task_type == "binary":
                loss = self.binary_classification_loss(pred, target)
            elif spec.task_type == "multiclass":
                loss = self.multiclass_onehot_loss(pred, target)
            else:
                raise ValueError(f"Unknown task type: {spec.task_type}")

            per_task[f"loss/{spec.name}"] = loss
            total = total + spec.weight * loss

        per_task["loss/total"] = total
        return total, per_task

    # --- Lightning hooks ---

    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x, y = batch
        outputs = self.forward(x)
        total, losses = self.compute_total_loss(outputs, y)

        self.log("train/loss", losses["loss/total"], prog_bar=True, on_step=True, on_epoch=True)
        for k, v in losses.items():
            if k != "loss/total":
                self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> None:
        x, y = batch
        outputs = self.forward(x)
        _, losses = self.compute_total_loss(outputs, y)
        self.log("val/loss", losses["loss/total"], prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> None:
        x, y = batch
        outputs = self.forward(x)
        _, losses = self.compute_total_loss(outputs, y)
        self.log("test/loss", losses["loss/total"], prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        x, _ = batch
        z = self.encode(x)
        outputs = self.forward(x)
        return {"latent": z, **{f"pred/{k}": v for k, v in outputs.items()}}

    def configure_optimizers(self) -> Dict[str, Any]:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-objective Lightning training example")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB logging (requires wandb)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    pl.seed_everything(args.seed, workers=True)

    # Example task configuration (arbitrary number of tasks supported)
    task_specs = [
        TaskSpec(name="y_reg", task_type="regression", output_dim=1, weight=1.0),
        TaskSpec(name="y_bin", task_type="binary", output_dim=1, weight=1.0),
        TaskSpec(name="y_mc", task_type="multiclass", output_dim=5, weight=1.0),
    ]

    data = MultiTaskDataModule(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        task_specs=task_specs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = MultiObjectiveModel(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        task_specs=task_specs,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    # Callbacks
    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="multiobjective-{epoch:02d}-{val_loss:.4f}",
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=8,
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")

    # Loggers
    tb_logger = TensorBoardLogger(save_dir="logs", name="multiobjective")
    wandb_logger = None
    if args.use_wandb:
        if WandbLogger is None:
            raise RuntimeError(
                "WandbLogger requested but not available. Install with: pip install wandb"
            )
        wandb_logger = WandbLogger(project="bes-ml-data", name="multiobjective")

    loggers = [tb_logger] + ([wandb_logger] if wandb_logger is not None else [])

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=loggers,
        callbacks=[ckpt, early, lrmon],
        enable_checkpointing=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data, ckpt_path="best")
    trainer.predict(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    main()
