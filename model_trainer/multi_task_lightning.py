import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torchmetrics

@dataclass
class MultiTaskDataset(Dataset):
    num_samples: int
    input_dim: int
    task_specs: List[Dict[str, Any]]

    def __post_init__(self):
        """
        Synthetic dataset for multi-task learning.
        """
        # Generate synthetic data
        self.inputs: torch.Tensor = torch.randn(self.num_samples, self.input_dim)
        self.targets: Dict[str, torch.Tensor] = {}
        
        for task in self.task_specs:
            if task['type'] == 'regression':
                self.targets[task['name']] = torch.randn(self.num_samples, task['output_dim'])
            elif task['type'] == 'binary_classification':
                self.targets[task['name']] = torch.randint(0, 2, (self.num_samples, task['output_dim'])).float()
            elif task['type'] == 'multiclass_classification':
                # Generate one-hot encoded targets
                num_classes: int = task['output_dim']
                indices: torch.Tensor = torch.randint(0, num_classes, (self.num_samples,))
                self.targets[task['name']] = F.one_hot(indices, num_classes=num_classes).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item_targets = {name: targets[idx] for name, targets in self.targets.items()}
        return self.inputs[idx], item_targets

@dataclass
class MultiTaskDataModule(pl.LightningDataModule):
    num_samples: int = 1000
    input_dim: int = 20
    task_specs: List[Dict[str, Any]] = None
    batch_size: int = 32
    num_workers: int = 0

    def __post_init__(self):
        super().__init__()

    def setup(self, stage: Optional[str] = None):
        # Create full dataset
        full_dataset: MultiTaskDataset = MultiTaskDataset(self.num_samples, self.input_dim, self.task_specs)
        
        # Split into train, val, test
        train_size: int = int(0.7 * len(full_dataset))
        val_size: int = int(0.15 * len(full_dataset))
        test_size: int = len(full_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

@dataclass(eq=False)
class MultiTaskModel(pl.LightningModule):
    input_dim: int
    latent_dim: int
    task_specs: List[Dict[str, Any]]
    learning_rate: float = 1e-3
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])

    def __post_init__(self):
        """
        Multi-objective neural network with shared latent space.
        """
        super().__init__()
        self.save_hyperparameters()

        # --- Shared Backbone (Encoder) ---
        layers: List[nn.Module] = []
        in_dim: int = self.input_dim
        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.latent_dim))
        layers.append(nn.ReLU()) # Activation for latent space
        self.backbone: nn.Sequential = nn.Sequential(*layers)

        # --- Task Heads ---
        self.heads: nn.ModuleDict = nn.ModuleDict()
        for task in self.task_specs:
            self.heads[task['name']] = self._build_head(self.latent_dim, task['output_dim'])
            
        # --- Loss Functions ---
        # We can store weights for each task loss if needed
        self.loss_weights: Dict[str, float] = {task['name']: task.get('weight', 1.0) for task in self.task_specs}

        # --- Metrics ---
        self.train_metrics: nn.ModuleDict = nn.ModuleDict()
        self.val_metrics: nn.ModuleDict = nn.ModuleDict()
        self.test_metrics: nn.ModuleDict = nn.ModuleDict()

        for task in self.task_specs:
            name: str = task['name']
            task_type: str = task['type']
            
            metrics: Optional[torchmetrics.MetricCollection] = None
            if task_type == 'regression':
                metrics = torchmetrics.MetricCollection({
                    'mse': torchmetrics.MeanSquaredError(),
                    'mae': torchmetrics.MeanAbsoluteError()
                })
            elif task_type == 'binary_classification':
                metrics = torchmetrics.MetricCollection({
                    'acc': torchmetrics.Accuracy(task='binary'),
                    'f1': torchmetrics.F1Score(task='binary')
                })
            elif task_type == 'multiclass_classification':
                num_classes: int = task['output_dim']
                metrics = torchmetrics.MetricCollection({
                    'acc': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
                    'f1': torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
                })
            else:
                continue
            
            self.train_metrics[name] = metrics.clone(prefix=f"train_{name}_")
            self.val_metrics[name] = metrics.clone(prefix=f"val_{name}_")
            self.test_metrics[name] = metrics.clone(prefix=f"test_{name}_")

    def _build_head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """Builds a simple MLP head for a task."""
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared encoding
        z = self.backbone(x)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(z)
        return outputs

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], log_prefix: str = "train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
        losses: Dict[str, torch.Tensor] = {}

        for task in self.task_specs:
            name: str = task['name']
            pred: torch.Tensor = outputs[name]
            target: torch.Tensor = targets[name]
            task_type: str = task['type']
            weight: float = self.loss_weights[name]

            loss: torch.Tensor
            if task_type == 'regression':
                loss = F.mse_loss(pred, target)
            elif task_type == 'binary_classification':
                loss = F.binary_cross_entropy_with_logits(pred, target)
            elif task_type == 'multiclass_classification':
                # Assuming one-hot targets, we use cross_entropy which supports soft targets 
                # or we convert to class indices.
                # PyTorch CrossEntropyLoss supports (N, C) logits and (N, C) probabilities (soft labels)
                loss = F.cross_entropy(pred, target)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            losses[f"{log_prefix}_{name}_loss"] = loss
            total_loss += weight * loss
        
        losses[f"{log_prefix}_total_loss"] = total_loss
        return total_loss, losses

    def _update_log_metrics(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], metric_dict: nn.ModuleDict) -> None:
        for task in self.task_specs:
            name: str = task['name']
            if name not in metric_dict:
                continue
                
            pred: torch.Tensor = outputs[name]
            target: torch.Tensor = targets[name]
            task_type: str = task['type']

            if task_type == 'multiclass_classification':
                # Convert one-hot to indices for metrics
                target_indices: torch.Tensor = torch.argmax(target, dim=1)
                metric_dict[name](pred, target_indices)
            else:
                metric_dict[name](pred, target)
            
            self.log_dict(metric_dict[name], on_step=False, on_epoch=True)

    def training_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x, targets = batch
        outputs = self(x)
        total_loss, losses = self._compute_loss(outputs, targets, log_prefix="train")
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        self._update_log_metrics(outputs, targets, self.train_metrics)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x, targets = batch
        outputs = self(x)
        total_loss, losses = self._compute_loss(outputs, targets, log_prefix="val")
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        self._update_log_metrics(outputs, targets, self.val_metrics)
        return total_loss

    def test_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        x, targets = batch
        outputs = self(x)
        total_loss, losses = self._compute_loss(outputs, targets, log_prefix="test")
        self.log_dict(losses, on_step=False, on_epoch=True)
        self._update_log_metrics(outputs, targets, self.test_metrics)
        return total_loss
    
    def predict_step(self, batch: Tuple[torch.Tensor, Dict[str, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss"
            }
        }

def main():
    # --- Configuration ---
    INPUT_DIM: int = 20
    LATENT_DIM: int = 16
    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 10
    
    # Define tasks
    # Note: For multiclass, output_dim = number of classes
    TASK_SPECS: List[Dict[str, Any]] = [
        {'name': 'signal_regression', 'type': 'regression', 'output_dim': 1, 'weight': 1.0},
        {'name': 'elm_detection', 'type': 'binary_classification', 'output_dim': 1, 'weight': 0.5},
        {'name': 'mode_classification', 'type': 'multiclass_classification', 'output_dim': 4, 'weight': 0.8}
    ]

    # --- Data ---
    data_module: MultiTaskDataModule = MultiTaskDataModule(
        num_samples=5000,
        input_dim=INPUT_DIM,
        task_specs=TASK_SPECS,
        batch_size=BATCH_SIZE
    )

    # --- Model ---
    model: MultiTaskModel = MultiTaskModel(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        task_specs=TASK_SPECS,
        learning_rate=1e-3
    )

    # --- Callbacks ---
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath='checkpoints',
        filename='multitask-{epoch:02d}-{val_total_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stopping: EarlyStopping = EarlyStopping(
        monitor='val_total_loss',
        patience=10,
        mode='min'
    )
    
    lr_monitor: LearningRateMonitor = LearningRateMonitor(logging_interval='epoch')

    # --- Loggers ---
    # Note: You might need to install wandb: pip install wandb
    # and tensorboard: pip install tensorboard
    tb_logger: TensorBoardLogger = TensorBoardLogger("logs", name="multitask_model")
    # wandb_logger = WandbLogger(project="bes_multitask", name="run_1") # Uncomment if wandb is configured

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=[tb_logger], # Add wandb_logger here if used
        accelerator="auto", # Auto-detect GPU/CPU
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=10
    )

    # --- Execution ---
    print("Starting Training...")
    trainer.fit(model, datamodule=data_module)

    print("Starting Testing...")
    trainer.test(model, datamodule=data_module)
    
    print("Starting Prediction...")
    predictions: List[Dict[str, torch.Tensor]] = trainer.predict(model, datamodule=data_module)
    print(f"Generated predictions for {len(predictions)} batches.")

if __name__ == '__main__':
    main()
