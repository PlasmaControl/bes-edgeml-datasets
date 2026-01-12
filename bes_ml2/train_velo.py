from __future__ import annotations
import os
from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
import shutil
import numpy as np

import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.profilers import PyTorchProfiler
import wandb

import psutil

try:
    from . import velocimetry_datamodule
    from . import elm_lightning_model
except:
    from bes_ml2 import velocimetry_datamodule
    from bes_ml2 import elm_lightning_model


@dataclasses.dataclass(eq=False)
class BES_Trainer:
    lightning_model: elm_lightning_model.Lightning_Model
    datamodule: velocimetry_datamodule.Velocimetry_Datamodule
    experiment_dir: str = './experiment_default'
    trial_name: str = None  # if None, use default Tensorboard scheme
    log_freq: int = 100
    wandb_log: bool = False
    num_train_batches: int = None
    num_val_batches: int = None
    num_predict_batches: int = None
    val_check_interval: int = 1.0

    def __post_init__(self):

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        if not self.trial_name:
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

        # **Set the model's log_dir to the experiment_dir**
        self.lightning_model.log_dir = str(self.experiment_dir)

        # set loggers
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
        gradient_clip_value: int = None,
        float_precision: str|int = '16-mixed' if torch.cuda.is_available() else 32,
    ):
        self.lightning_model.log_dir = self.datamodule.log_dir = self.trial_dir
        monitor_metric = self.lightning_model.monitor_metric
        metric_mode = 'min' if 'loss' in monitor_metric else 'max'
        # torch.set_float32_matmul_precision('medium')
        torch.set_float32_matmul_precision('high')  # stricter, no TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # set callbacks
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor=monitor_metric,
                mode=metric_mode,
                save_last=True,
            ),
            EarlyStopping(
                monitor=monitor_metric,
                mode=metric_mode,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience,
                log_rank_zero_only=True,
                verbose=True,
            ),
        ]

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
            if self.datamodule.split_train_data_per_gpu:
                del self.datamodule._train_dataloader  
            else: 
                del self.datamodule.datasets['train']

            del self.datamodule.datasets['validation']
            trainer.test(datamodule=self.datamodule, ckpt_path='best')

        if skip_predict is False:
            # free up space
            if self.datamodule.split_train_data_per_gpu:
                del self.datamodule._train_dataloader  
            elif skip_test is False: 
                del self.datamodule.datasets['test']
                
            trainer.predict(datamodule=self.datamodule, ckpt_path='best')

        self.last_model_path = Path(trainer.checkpoint_callback.last_model_path).absolute()
        print(f"Last model path: {self.last_model_path}")
        best_model_path = Path(trainer.checkpoint_callback.best_model_path).absolute()
        self.best_model_path = best_model_path.parent/'best.ckpt'
        shutil.copyfile(
            src=best_model_path,
            dst=self.best_model_path,
        )
        print(f"Best model path: {self.best_model_path}")

if __name__=='__main__':

    checkpoint = None

    if checkpoint:
        # load data and model from checkpoint
        lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(checkpoint_path=checkpoint)
        datamodule = velocimetry_datamodule.Velocimetry_Datamodule.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        block_cols = [1, 3, 5, 7]
        row_stride   = 1
        row_offset   = 4
        R_sel = len(np.arange(8)[row_offset::row_stride])
        C_sel = len(block_cols) 
        # initiate new data and model
        lightning_model = elm_lightning_model.Lightning_Model(
            encoder_lr=1e-3,
            decoder_lr=1e-3,
            signal_window_size=48,
            n_rows=R_sel,
            n_cols=C_sel,
            monitor_metric='sum_loss/val',
            lr_scheduler_threshold=1e-3,
            lr_scheduler_patience=10,
            weight_decay=0.001,
            encoder_type='none',
            cnn_nlayers=2,
            cnn_num_kernels=[16, 32],
            cnn_kernel_time_size=[8, 4],
            cnn_kernel_spatial_size=[3, 3],
            cnn_padding = [1, 1],
            cnn_maxpool_spatial_size = [2, 1],
            cnn_maxpool_time_size = [2, 2],
            leaky_relu_slope=0.001,
            mlp_layers=(50, 50),
            mlp_dropout=0.1,
            velocimetry_mlp=True,
        )

        datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
            data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20250824_psi_interp.hdf5',
            signal_window_size=lightning_model.signal_window_size,
            batch_size=256,
            num_workers=1,
            seed=0,
            world_size=1,
            lower_cutoff_frequency_hz=60e3,
            upper_cutoff_frequency_hz=150e3,  # Upper cutoff frequency in Hz
            start_time_ms=2400,
            split_method='shot',
            fraction_validation=0.1,
            fraction_test=0.05,
            train_shots=['145384', '145391', '145410', '145420', '145422', '145427', '157303', '157322', '157372', '157374', '158076', '189189', '203659', '203663'],
            validation_shots=['145388', '145419', '157376', '200635',  '203671'],
            test_shots=['145387', '145425', '157323', '157373', '157375', '157377', '159443', '189191', '189199', '200021', '203660', '203672', '203665', '203667'],
            predict_shots=['145384', '145388', '145391', '145419', '145425', '145385', '145422', '145410', '157373', '145387',  '145420', '145427', '159443', '200635'],
            split_train_data_per_gpu=True,
            do_flip_augmentation=True,
            block_cols=block_cols,
            row_stride=row_stride,
            row_offset=row_offset,
            target_sampling_hz=1_000_000.0,
            label_target_psi=0.9,
            label_tolerance_ms=0.6,
            window_hop=1,
            # --- CRITICAL: keep these consistent with the model ---
            n_rows=R_sel,   # 8 with your settings
            n_cols=C_sel,   # 4 with ('last',4)
        )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        wandb_log=False,
        num_train_batches=1,
        num_val_batches=1,
    )

    trainer.run_all(
        max_epochs=1,
        # skip_test=True,
        # skip_predict=True,
    )
