from __future__ import annotations

from typing import Any, Optional, Union
import os
from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
import shutil
import numpy as np
import importlib

import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy

"""Training entrypoint for BES velocimetry experiments.

This file wires together the Lightning model + datamodule, configures loggers/callbacks,
and writes artifacts into an experiment directory. It is intended to work both when
executed as part of the `bes_ml2` package and when run as a standalone script.
"""

# Lightning changed where ModelSummary is exported across versions; import dynamically
# for compatibility instead of pinning a single import path.
try:
    ModelSummary: Any = getattr(
        importlib.import_module("lightning.pytorch.utilities.model_summary.model_summary"),
        "ModelSummary",
    )
except Exception:
    ModelSummary = getattr(
        importlib.import_module("lightning.pytorch.utilities.model_summary"),
        "ModelSummary",
    )

# W&B is an optional dependency; keep import-time failures from breaking non-W&B runs.
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

try:
    from bes_ml2 import velocimetry_datamodule
    from bes_ml2.elm_lightning_model import Lightning_Model
except ImportError:
    from . import velocimetry_datamodule
    from .elm_lightning_model import Lightning_Model


@dataclasses.dataclass(eq=False)
class BES_Trainer:
    """Small convenience wrapper around Lightning's `Trainer`.

    The goal is to centralize:
    - experiment directory naming
    - logger/callback setup
    - checkpoint bookkeeping (best/last)
    """
    lightning_model: Lightning_Model
    datamodule: velocimetry_datamodule.Velocimetry_Datamodule
    experiment_dir: Union[str, Path] = './experiment_default'
    trial_name: Optional[str] = None  # if None, use default Tensorboard scheme
    log_freq: int = 100
    wandb_log: bool = False
    num_train_batches: Optional[int] = None
    num_val_batches: Optional[int] = None
    num_predict_batches: Optional[int] = None
    val_check_interval: float = 1.0

    # Derived/filled in during `__post_init__` / `run_all`.
    experiment_name: str = dataclasses.field(init=False)
    experiment_parent_dir: Path = dataclasses.field(init=False)
    trial_dir: Path = dataclasses.field(init=False)
    loggers: list[Any] = dataclasses.field(init=False)
    last_model_path: Path = dataclasses.field(init=False)
    best_model_path: Path = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # Print config early so SLURM logs capture the exact run settings.
        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        # Only print user-configurable (init=True) fields here.
        # Derived init=False fields are populated later in this method / in run_all.
        for field in dataclasses.fields(self):
            if not field.init:
                continue
            field_name = field.name
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if default_value is not dataclasses.MISSING and value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        if not self.trial_name:
            # Default: timestamp-based run directory.
            # If available, incorporate a SLURM/launcher-provided identifier to avoid collisions.
            datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
            if slurm_identifier:
                self.trial_name = f"r{slurm_identifier}_{datetime_str}"
            else:
                self.trial_name = f"r{datetime_str}"

        self.experiment_dir = Path(self.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = self.experiment_dir.name
        self.experiment_parent_dir = self.experiment_dir.parent

        # Set the model's log_dir to the experiment_dir so any model-side artifact writing
        # (plots, tables, etc.) ends up alongside the Lightning logs.
        self.lightning_model.log_dir = str(self.experiment_dir)

        # TensorBoard logger defines the canonical trial directory.
        tb_logger = TensorBoardLogger(
            save_dir=self.experiment_parent_dir,
            name=self.experiment_name,
            version=self.trial_name,
            default_hp_metric=False,
        )
        self.trial_dir = Path(tb_logger.log_dir).absolute()
        print(f"Trial directory: {self.trial_dir}")
        self.loggers = [tb_logger]

        if self.wandb_log:
            # W&B is optional; avoid importing/initializing it for users who do not want it.
            if wandb is None:
                raise RuntimeError(
                    "wandb_log=True, but `wandb` is not importable. "
                    "Install wandb (and the Lightning Wandb logger extras) or set wandb_log=False."
                )
            wandb.login()
            wandb_logger = WandbLogger(
                save_dir=self.experiment_dir,
                project=self.experiment_name,
                name=self.trial_name,
            )
            wandb_logger.watch(
                self.lightning_model, 
                log='all', 
                log_freq=self.log_freq,
            )
            self.loggers.append(wandb_logger)

        # Print a model summary once at startup (we disable Lightning's built-in summary
        # because it can be noisy / version-dependent in DDP settings).
        print("Model Summary:")
        print(ModelSummary(self.lightning_model, max_depth=-1))
        self.lightning_model.log_dir = self.datamodule.log_dir = self.trial_dir

    def run_all(
        self,
        max_epochs: int = 2,
        skip_test: bool = False,
        skip_predict: bool = False,
        early_stopping_min_delta: float = 1e-3,
        early_stopping_patience: int = 50,
        gradient_clip_value: Optional[float] = None,
        float_precision: Union[str, int] = '16-mixed' if torch.cuda.is_available() else 32,
    ) -> None:
        self.lightning_model.log_dir = self.datamodule.log_dir = self.trial_dir
        monitor_metric = self.lightning_model.monitor_metric
        metric_mode = 'min' if 'loss' in monitor_metric else 'max'
        # Numerical consistency knobs:
        # - `set_float32_matmul_precision('high')` requests stricter matmul kernels.
        # - TF32 is disabled explicitly for determinism/reproducibility.
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # set callbacks
        checkpoint_cb = ModelCheckpoint(
            monitor=monitor_metric,
            mode=metric_mode,
            save_last=True,
        )
        callbacks = [
            LearningRateMonitor(),
            checkpoint_cb,
            EarlyStopping(
                monitor=monitor_metric,
                mode=metric_mode,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience,
                log_rank_zero_only=True,
                verbose=True,
            ),
        ]

        # DDP's `find_unused_parameters` must be enabled if some model branches are
        # disabled (e.g., only a subset of heads are active).
        frontends_active = [value for value in self.lightning_model.frontends_active.values()]
        some_unused = False in frontends_active

        trainer = Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_value,
            logger=self.loggers,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=False,
            log_every_n_steps=self.log_freq,
            num_nodes=int(os.getenv('SLURM_NNODES', default=1)),
            precision=float_precision,
            strategy=DDPStrategy(find_unused_parameters=some_unused, timeout=timedelta(seconds=9600)),
            use_distributed_sampler=False,
            # profiler=PyTorchProfiler(),
            limit_train_batches=self.num_train_batches,
            limit_val_batches=self.num_val_batches,
            limit_predict_batches=self.num_predict_batches,
            val_check_interval=self.val_check_interval,
        )
        self.datamodule.is_global_zero = trainer.is_global_zero
        if trainer.is_global_zero:
            self.trial_dir.mkdir(parents=True, exist_ok=True)

        trainer.fit(
            self.lightning_model, 
            datamodule=self.datamodule,
        )
        
        if skip_test is False:
            # Free memory from train/validation datasets before testing.
            # This matters when datasets are large and the job is memory-bound.
            if self.datamodule.split_train_data_per_gpu:
                if hasattr(self.datamodule, "_train_dataloader"):
                    del self.datamodule._train_dataloader
            else:
                self.datamodule.datasets.pop('train', None)

            self.datamodule.datasets.pop('validation', None)
            trainer.test(datamodule=self.datamodule, ckpt_path='best')

        if skip_predict is False:
            # Free up space again before prediction.
            if self.datamodule.split_train_data_per_gpu:
                if hasattr(self.datamodule, "_train_dataloader"):
                    del self.datamodule._train_dataloader
            elif skip_test is False:
                self.datamodule.datasets.pop('test', None)
                
            trainer.predict(datamodule=self.datamodule, ckpt_path='best')

        # Record both "last" (end-of-fit) and "best" (monitored metric) checkpoints.
        self.last_model_path = Path(checkpoint_cb.last_model_path).absolute()
        print(f"Last model path: {self.last_model_path}")
        best_model_path = Path(checkpoint_cb.best_model_path).absolute()
        self.best_model_path = best_model_path.parent/'best.ckpt'
        shutil.copyfile(
            src=best_model_path,
            dst=self.best_model_path,
        )
        print(f"Best model path: {self.best_model_path}")

if __name__=='__main__':

    logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
    world_size = int(os.getenv('SLURM_NTASKS', 1))
    world_rank = int(os.getenv('SLURM_PROCID', 0))

    block_cols = [1, 3, 5, 7]
    row_stride   = 1
    row_offset   = 4
    R_sel = len(np.arange(8)[row_offset::row_stride])
    C_sel = len(block_cols)

    good_times_psi_93 = {
        # 145384: [()], # this means use all times
        # 145387: [()],
        # 145388: [()],
        145391: [(1900, 4200)],
        # 145410: [()],
        # 145419: [()],
        # 145420: [()],
        # 145422: [()],
        # 145425: [()],
        # 145427: [()],
        # 157303: [()],
        # 157322: [()],
        # 157323: [()],
        # 157372: [()],
        # 157373: [()],
        # 157374: [()],
        157375: [(1900, 3600), (4000, 5500)],
        # 157376: [()],
        157377: [(1800, 5000), (5400, 6000)],
        # 158076: [()],
        159443: [(1900, 5600)],
        189189: [(2000, 4700)],
        # 189191: [()],
        189199: [(1800, 3000), (4200, 4700)],
        # 200021: [()],
        # 200632: [()],
        # 200634: [()],
        200637: [(1100, 3800)],
        200638: [(800, 3500)],
        200639: [(800, 4600)],
        200643: [(800, 4600)],
        # 203152: [()],
        # 203416: [()],
        203417: [(2250, 2700), (2900, 3300), (3600, 4100)],
        # 203418: [()],
        203419: [(2250, 3500), (3750, 4000)],
        # 203420: [()],
        203423: [(2300, 3850)],
        203469: [(600, 4600)],
        203470: [(500, 3900)],
        203471: [(500, 3800)],
        # 203475: [()],
        203483: [(800, 4300)],
        203484: [(800, 4200)],
        203485: [(800, 2100), (3200, 4500)],
        # 203659: [()],
        203660: [(1100, 2900)],
        # 203662: [()],
        203663: [(1100, 3900)],
        203664: [(1000, 4000)],
        # 203665: [()],
        # 203667: [()],
        # 203671: [()],
        # 203672: [()],
        203946: [(4000, 5400)],
        204286: [(1800, 4600)],
        204287: [(1500, 4000)],
        # 204288: [()],
        204289: [(1100, 4300)],
        204290: [(1100, 4100)],
        204291: [(1100, 3700)],
        # 204292: [()],
        # 204293: [()],
        204294: [(1100, 4100)],
        # 204295: [()],
        # 204296: [()],
        # 204297: [()],
        204299: [(1500, 4500)],
        # 204301: [()],
        204302: [(1800, 4100)],
        204303: [(1600, 4400)],
        # 204837: [()],
        205867: [(2200, 4400)],

    }

    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
            data_file='/global/cfs/cdirs/m3586/kgill/velocimetry_data/20251027_raw_signals_psi_interp.hdf5',
            signal_window_size=48,
            batch_size=1024,
            num_workers=4,
            seed=0,
            world_size=world_size,
            standardize_signals=False,
            split_method='shot',
            train_shots = [
                # 145xxx
                '145384', '145420', '145425',
                # 157xxx 
                '157303', '157372', '157375', '158076',
                # 200xxx
                '200643',
                # Early 203xxx 
                '203416', '203420', '203483',
                # Late 203xxx + 204xxx
                '203664', '203671', '204292', '204837',
            ],
            validation_shots = [
                '145388', '145419',      # 145xxx
                '157322', '157373',      # 157xxx  
                '203417', '203423',      # early 203xxx
                '203665', '203672',      # late 203xxx
            ],            
            test_shots = ['145387', '145391', '145410', '145422', '145427', '157323', '157374', '157377', '159443', '189189', '189191', '189199', '200634', '200637', '200638', '200639', '203152', '203418', '203419', '203469', '203470', '203471', '203484', '203659', '203660', '203663', '203667', '203946', '204286', '204287', '204288', '204289', '204290', '204291', '204294',  '204296', '204297', '204299', '204301', '204302', '204303'],
            predict_shots = ['145384', '145420', '145425', '157303', '157372', '157375', '158076', '200643', '203416', '203420', '203483', '203664', '203671', '204292', '204295', '204837', '145388', '145419', '157322', '157373', '157376', '203417', '203423', '203475', '203485', '203665', '203672', '204293', '145387', '145391', '145410', '145422', '145427', '157323', '157374', '157377', '159443', '189189', '189191', '189199', '200634', '200637', '200638', '200639', '203152', '203418', '203419', '203469', '203470', '203471', '203484', '203659', '203660', '203663', '203667', '203946', '204286', '204287', '204288', '204289', '204290', '204291', '204294',  '204296', '204297', '204299', '204301', '204302', '204303'],
            split_train_data_per_gpu=True,
            do_flip_augmentation=True, # crucial
            shot_time_windows = good_times_psi_93, # good times
            # --- block + label knobs ---
            block_cols=block_cols,
            row_stride=row_stride,
            row_offset=row_offset,
            target_sampling_hz=1_000_000.0,
            label_target_psi=0.93, # 0.85, 0.88, 0.91, 0.93, 0.95
            label_tolerance_ms=0.6,
            window_hop=1,
            n_rows=R_sel,   
            n_cols=C_sel,   
    )

    weight_decay = 0.00001
    trial_name = f'{logger_hash}'

    lightning_model = Lightning_Model(
        encoder_lr=1e-3,
        decoder_lr=1e-3,
        signal_window_size=datamodule.signal_window_size,
        n_rows=R_sel,
        n_cols=C_sel,
        monitor_metric='sum_loss/val',
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=5,
        weight_decay=weight_decay,
        encoder_type='none',
        mlp_layers=(50, 50),
        mlp_dropout=0.001,
        velocimetry_mlp=True,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
    )

    trainer.run_all(
        max_epochs=40,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=100,
        skip_test=False,
        skip_predict=False,
        float_precision=32,
    )

    # checkpoint = None

    # if checkpoint:
    #     # load data and model from checkpoint
    #     lightning_model = Lightning_Model.load_from_checkpoint(checkpoint_path=checkpoint)
    #     datamodule = velocimetry_datamodule.Velocimetry_Datamodule.load_from_checkpoint(checkpoint_path=checkpoint)
    # else:
    #     block_cols = [1, 3, 5, 7]
    #     row_stride   = 1
    #     row_offset   = 4
    #     R_sel = len(np.arange(8)[row_offset::row_stride])
    #     C_sel = len(block_cols) 
    #     # initiate new data and model
    #     lightning_model = Lightning_Model(
    #         encoder_lr=1e-3,
    #         decoder_lr=1e-3,
    #         signal_window_size=48,
    #         n_rows=R_sel,
    #         n_cols=C_sel,
    #         monitor_metric='sum_loss/val',
    #         lr_scheduler_threshold=1e-3,
    #         lr_scheduler_patience=10,
    #         weight_decay=0.001,
    #         encoder_type='none',
    #         cnn_nlayers=2,
    #         cnn_num_kernels=[16, 32],
    #         cnn_kernel_time_size=[8, 4],
    #         cnn_kernel_spatial_size=[3, 3],
    #         cnn_padding = [1, 1],
    #         cnn_maxpool_spatial_size = [2, 1],
    #         cnn_maxpool_time_size = [2, 2],
    #         leaky_relu_slope=0.001,
    #         mlp_layers=(50, 50),
    #         mlp_dropout=0.1,
    #         velocimetry_mlp=True,
    #     )

    #     datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
    #         # data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20250824_psi_interp.hdf5',
    #         data_file='/global/homes/d/drsmith/scratch-ml/data/20251027_raw_signals_psi_interp.hdf5',
    #         signal_window_size=lightning_model.signal_window_size,
    #         batch_size=256,
    #         num_workers=1,
    #         seed=0,
    #         world_size=1,
    #         lower_cutoff_frequency_hz=60e3,
    #         upper_cutoff_frequency_hz=150e3,  # Upper cutoff frequency in Hz
    #         start_time_ms=2400,
    #         split_method='shot',
    #         fraction_validation=0.1,
    #         fraction_test=0.05,
    #         train_shots=['145384', '145391', '145410', '145420', '145422', '145427', '157303', '157322', '157372', '157374', '158076', '189189', '203659', '203663'],
    #         validation_shots=['145388', '145419', '157376', '200635',  '203671'],
    #         test_shots=['145387', '145425', '157323', '157373', '157375', '157377', '159443', '189191', '189199', '200021', '203660', '203672', '203665', '203667'],
    #         predict_shots=['145384', '145388', '145391', '145419', '145425', '145385', '145422', '145410', '157373', '145387',  '145420', '145427', '159443', '200635'],
    #         split_train_data_per_gpu=True,
    #         do_flip_augmentation=True,
    #         block_cols=block_cols,
    #         row_stride=row_stride,
    #         row_offset=row_offset,
    #         target_sampling_hz=1_000_000.0,
    #         label_target_psi=0.9,
    #         label_tolerance_ms=0.6,
    #         window_hop=1,
    #         # --- CRITICAL: keep these consistent with the model ---
    #         n_rows=R_sel,   # 8 with your settings
    #         n_cols=C_sel,   # 4 with ('last',4)
    #     )

    # trainer = BES_Trainer(
    #     lightning_model=lightning_model,
    #     datamodule=datamodule,
    #     experiment_dir='./exp_gill01',
    #     wandb_log=False,
    #     num_train_batches=1,
    #     num_val_batches=1,
    # )

    # trainer.run_all(
    #     max_epochs=1,
    #     skip_test=True,
    #     skip_predict=True,
    # )
