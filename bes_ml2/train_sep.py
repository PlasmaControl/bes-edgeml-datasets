from __future__ import annotations
import os
from pathlib import Path
import dataclasses
from datetime import datetime, timedelta
import shutil

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
    from . import separatrix_datamodule
    from . import elm_lightning_model
except:
    from bes_ml2 import separatrix_datamodule
    from bes_ml2 import elm_lightning_model


@dataclasses.dataclass(eq=False)
class BES_Trainer:
    lightning_model: elm_lightning_model.Lightning_Model
    datamodule: separatrix_datamodule.Separatrix_Datamodule
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

    def run_all(
        self,
        max_epochs: int = 2,
        skip_test: bool = False,
        skip_predict: bool = False,
        early_stopping_min_delta: float = 1e-3,
        early_stopping_patience: int = 50,
        gradient_clip_value: int = None,
        float_precision: str|int = '16-mixed' if torch.cuda.is_available() else 32,
        debug_predict: bool = False,
        visualize_post_test: bool = False,
    ):
        self.lightning_model.log_dir = self.datamodule.log_dir = self.trial_dir
        monitor_metric = self.lightning_model.monitor_metric
        metric_mode = 'min' if 'loss' in monitor_metric else 'max'
        torch.set_float32_matmul_precision('medium')

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
            trainer.test(datamodule=self.datamodule, ckpt_path='best')
            if visualize_post_test:
                # Visualization after testing
                self.visualize_post_test(trainer, num_top_freq_indices=3, subwindow_idx=0)

        if skip_predict is False:
            # free up space
            del self.datamodule._train_dataloader
            del self.datamodule.datasets['validation']
            
            self.lightning_model.manual_predict_separatrix(
                self.datamodule.test_dataloader(), 
                self.datamodule.datasets['test'],
                debug=debug_predict,
                save_filename=None,
                plot_inference=False,
            )

        self.last_model_path = Path(trainer.checkpoint_callback.last_model_path).absolute()
        print(f"Last model path: {self.last_model_path}")
        best_model_path = Path(trainer.checkpoint_callback.best_model_path).absolute()
        self.best_model_path = best_model_path.parent/'best.ckpt'
        shutil.copyfile(
            src=best_model_path,
            dst=self.best_model_path,
        )
        print(f"Best model path: {self.best_model_path}")

    def visualize_post_test(self, trainer, num_top_freq_indices, subwindow_idx):
        # Load a batch of data for visualization
        dataloader = self.datamodule.test_dataloader()
        batch = next(iter(dataloader))

        # Extract signals from the batch
        signals = batch[0]  # Assuming signals are the first element in the batch

        # Load the model from the checkpoint
        model = self.lightning_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # Call the visualization method
        model.visualize_fft_features(signals, num_top_freq_indices=num_top_freq_indices, subwindow_idx=subwindow_idx)


if __name__=='__main__':

    checkpoint = None

    if checkpoint:
        # load data and model from checkpoint
        lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(checkpoint_path=checkpoint)
        datamodule = separatrix_datamodule.Separatrix_Datamodule.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        # initiate new data and model
        cnn_nlayers = 1
        lightning_model = elm_lightning_model.Lightning_Model(
            encoder_lr=1e-3,
            decoder_lr=1e-5,
            signal_window_size=1,
            encoder_type='none',
            cnn_nlayers=cnn_nlayers,
            cnn_num_kernels=(10),
            cnn_kernel_time_size=[2] * cnn_nlayers,
            cnn_kernel_spatial_size=3,
            cnn_maxpool_time_size=(2),
            # cnn_padding="same",
            # cnn_maxpool_spatial_size=[(1, 2) for _ in range(cnn_nlayers)],
            cnn_maxpool_spatial_size=[(1, 1)],
            separatrix_mlp=True,
            velocimetry_mlp=False,
            reconstruction_decoder=False,
            multiclass_classifier_mlp=False,
            time_to_elm_mlp=False,
            classifier_mlp=False,
            mlp_layers=(60, 60),
            mlp_dropout=0.1,
            leaky_relu_slope=0.001,
            n_rows=8,
            n_cols=8,
        )
        datamodule = separatrix_datamodule.Separatrix_Datamodule(
            data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20241014.hdf5',
            seed=1,
            # max_shots_per_class=35,
            # max_shots=80,
            # lower_cutoff_frequency_hz=2.5e3,
            # upper_cutoff_frequency_hz=150e3,
            signal_window_size=lightning_model.signal_window_size,
            n_rows=lightning_model.n_rows,
            n_cols=lightning_model.n_cols,
            fraction_test=0.15,
            fraction_validation=0.15,
            batch_size=128,
            num_workers=0,
            plot_data_stats=False,
            world_size=1,
            # r_avg_bounds=(220,230),
            # z_avg_bounds=(-2,2),
            # delz_avg_bounds=(1,2.5),
            # r_avg_bounds_class_3=(200,240),
            # z_avg_bounds_class_3=(-6,6),
            # delz_avg_bounds_class_3=(0,6),
            # clip_signals=2.0,
            # force_validation_shots=['175490', '187035', '191376', '164869', '171471', '172212', '184463', '160778'],
            # force_test_shots=['164798'],
            # train_shots=['164793'],
            # test_shots=['164798'],
            # test_only=True,
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
