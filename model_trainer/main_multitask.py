from pathlib import Path
import dataclasses
from datetime import datetime
from types import NoneType
from typing import OrderedDict, Any, Sequence
import os
import time
import gc
import psutil

import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split 
import scipy.signal
import h5py
import wandb

import torch
import torch.nn
import torch.cuda
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data

from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import \
    LearningRateMonitor, EarlyStopping, ModelCheckpoint, BatchSizeFinder
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.combined_loader import CombinedLoader

torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)


def print_fields(obj):
    print(f"{obj.__class__.__name__} fields:")
    class_fields_dict = {field.name: field for field in dataclasses.fields(obj.__class__)}
    for field_name in dataclasses.asdict(obj):
        value = getattr(obj, field_name)
        field_str = f"  {field_name}: {value}"
        default_value = class_fields_dict[field_name].default
        if value != default_value:
            field_str += f" (default {default_value})"
        print(field_str)


@dataclasses.dataclass(eq=False)
class _Base_Class:
    signal_window_size: int = 1024
    is_global_zero: bool = False

    def __post_init__(self):
        assert np.log2(self.signal_window_size).is_integer(), \
            'Signal window must be power of 2'


@dataclasses.dataclass(eq=False)
class Model(LightningModule, _Base_Class):
    elm_classifier: bool = True
    conf_classifier: bool = False
    lr: float = 1e-3  # maximum LR used by first layer
    lr_scheduler_patience: int = 50
    lr_scheduler_threshold: float = 1e-3
    lr_warmup_epochs: int = 8
    # lr_layerwise_decrement: float = 1.
    weight_decay: float = 1e-6
    leaky_relu_slope: float = 2e-2
    monitor_metric: str = None
    use_optimizer: str = 'SGD'
    elm_mean_loss_factor: float = None
    conf_mean_loss_factor: float = None
    initial_weight_factor: float = 1.0
    # feature_batchnorm: bool = True
    # task_batchnorm: bool = False

    def __post_init__(self):

        # init superclasses
        super().__init__()
        super(LightningModule, self).__post_init__()

        self.save_hyperparameters()
        if self.is_global_zero:
            print_fields(self)

        # single input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.make_feature_model()

        # task sub-models and metrics
        self.task_models = torch.nn.ModuleDict()
        self.task_metrics: dict[str, dict] = {}

        # Sub-model: ELM median time-to-ELM binary classifier
        if self.elm_classifier:
            task_name = 'elm_classifier'
            self.zprint(f"Task {task_name}")
            self.task_models[task_name] = self.make_mlp_classifier()
            self.task_metrics[task_name] = {
                'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
                'f1_score': sklearn.metrics.f1_score,
                'precision_score': sklearn.metrics.precision_score,
                'recall_score': sklearn.metrics.recall_score,
                'mean_stat': lambda t: torch.abs(torch.mean(t)), #torch.mean,
                'std_stat': torch.std,
            }
            if self.monitor_metric is None:
                self.monitor_metric = f'{task_name}/f1_score/val'

        # sub-model: Confinement mode multi-class classifier
        if self.conf_classifier:
            task_name = 'conf_classifier'
            self.zprint(f"Task {task_name}")
            self.task_models[task_name] = self.make_mlp_classifier(n_out=4)
            self.task_metrics[task_name] = {
                'ce_loss': torch.nn.functional.cross_entropy,
                'f1_score': sklearn.metrics.f1_score,
                'precision_score': sklearn.metrics.precision_score,
                'recall_score': sklearn.metrics.recall_score,
                'mean_stat': lambda t: torch.abs(torch.mean(t)), #torch.mean,
                'std_stat': torch.std,
            }
            if self.monitor_metric is None:
                self.monitor_metric = f'{task_name}/f1_score/val'

        self.task_names = list(self.task_models.keys())

        if self.is_global_zero: 
            good_init = False
            while good_init == False:
                self.zprint("Initializing model to uniform random weights and biases=0")
                good_init = True
                for name, param in self.named_parameters():
                    if name.endswith("bias"):
                        self.zprint(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                        param.data.fill_(0)
                    elif name.endswith("weight"):
                        if 'BatchNorm' in name:
                            self.zprint(f"  {name}: initialized to ones (numel {param.data.numel()})")
                            param.data.fill_(1)
                        else:
                            n_in = np.prod(param.shape[1:])
                            sqrt_k =  np.sqrt(3*self.initial_weight_factor / n_in)
                            param.data.uniform_(-sqrt_k, sqrt_k)
                            # param.data.normal_(std=sqrt_k)
                            self.zprint(f"  {name}: initialized to normal +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")
                print("Batch evaluation (batch_size=256) with randn() data")
                batch_input = {
                    task: [torch.randn(
                        size=[256]+list(self.input_data_shape[1:]),
                        dtype=torch.float32,
                    )]
                    for task in self.task_names
                }
                batch_output = self(batch_input)
                for task, task_output in batch_output.items():
                    if good_init == False: continue
                    if task_output.mean().abs() / task_output.std() > 0.25:
                        good_init = False
                        continue
                    self.zprint(f"  Task {task} output shape: {task_output.shape}")
                    self.zprint(f"  Task {task} output mean {task_output.mean():.4f} stdev {task_output.std():.4f} min/max {task_output.min():.3f}/{task_output.max():.3f}")

        self.zprint(f"Total model parameters: {self.param_count(self):,d}")
        return

    @staticmethod
    def param_count(model: LightningModule) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def make_feature_model(self) -> None:

        self.zprint("Feature space sub-model")

        feature_layer_dict = OrderedDict()

        conv_layers = (
            {'out_channels': 6, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            {'out_channels': 8, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 10, 'kernel': (4, 1, 1), 'stride': (4, 1, 1)},
            {'out_channels': 16, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 16, 'kernel': (1, 3, 3), 'stride': 1},
            {'out_channels': 16, 'kernel': (1, 2, 2), 'stride': 1},
            {'out_channels': 16, 'kernel': (4, 1, 1), 'stride': (4, 1, 1)},
        )

        data_shape = self.input_data_shape
        self.zprint(f"  Input data shape: {data_shape}  (size {np.prod(data_shape)})")
        out_channels: int|Any = None
        for i_layer, layer in enumerate(conv_layers):
            conv_layer_name = f"L{i_layer:02d}_Conv"
            conv = torch.nn.Conv3d(
                in_channels=1 if out_channels is None else out_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
            )
            n_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            self.zprint(f"  {conv_layer_name} kern {conv.kernel_size}  stride {conv.stride}  out_ch {conv.out_channels}  param {n_params:,d}  output {data_shape} (size {np.prod(data_shape)})")
            out_channels = conv.out_channels
            if i_layer > 0:
                feature_layer_dict[f"L{i_layer:02d}_Dropout"] = torch.nn.Dropout3d(0.05)
            feature_layer_dict[conv_layer_name] = conv
            feature_layer_dict[f"L{i_layer:02d}_LeRu"] = torch.nn.LeakyReLU(self.leaky_relu_slope)
            feature_layer_dict[f"L{i_layer:02d}_BatchNorm"] = torch.nn.BatchNorm3d(out_channels)

        feature_layer_dict['Flatten'] = torch.nn.Flatten()
        self.feature_model = torch.nn.Sequential(feature_layer_dict)
        self.feature_space_size = self.feature_model(torch.zeros(self.input_data_shape)).numel()

        self.zprint(f"  Feature sub-model parameters: {self.param_count(self.feature_model):,d}")
        self.zprint(f"  Feature space size: {self.feature_space_size}")

    def make_mlp_classifier(self, n_out: int = 1) -> torch.nn.Module:

        self.zprint("MLP classifier sub-model")

        mlp_layer_dict = OrderedDict()

        assert self.feature_space_size
        mlp_layer_sizes = (self.feature_space_size, 64, 32, n_out)
        n_layers = len(mlp_layer_sizes)

        for i_layer in range(n_layers-1):
            mlp_layer_name = f"L{i_layer:02d}_FC"
            mlp_layer = torch.nn.Linear(
                in_features=mlp_layer_sizes[i_layer],
                out_features=mlp_layer_sizes[i_layer+1],
                bias=True if i_layer+1<n_layers-1 else False,
            )
            n_params = sum(p.numel() for p in mlp_layer.parameters() if p.requires_grad)
            self.zprint(f"  {mlp_layer_name}  in_features {mlp_layer.in_features}  out_features {mlp_layer.out_features}  parameters {n_params:,d}")
            if i_layer+1 < n_layers-1:
                mlp_layer_dict[f"L{i_layer:02d}_Dropout"] = torch.nn.Dropout1d(0.05)
            mlp_layer_dict[mlp_layer_name] = mlp_layer
            if i_layer+1 < n_layers-1:
                mlp_layer_dict[f"L{i_layer:02d}_LeRu"] = torch.nn.LeakyReLU(self.leaky_relu_slope)

        mlp_classifier = torch.nn.Sequential(mlp_layer_dict)

        self.zprint(f"  MLP sub-model parameters: {self.param_count(mlp_classifier):,d}")

        return mlp_classifier

    def configure_optimizers(self):
        self.zprint(f"Using {self.use_optimizer.upper()} optimizer")
        optim_kwargs = {
            'params': self.parameters(),
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
        if self.use_optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(momentum=0.2, **optim_kwargs)
        elif self.use_optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(**optim_kwargs)
        else:
            raise ValueError

        lr_reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            min_lr=1e-4,
            verbose=True,
        )
        lr_warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.1,
            total_iters=self.lr_warmup_epochs,
            verbose=True,
        )
        return_optim_list = [self.optimizer]
        return_lr_scheduler_list = [
            {'scheduler': lr_reduce_on_plateau, 'monitor': self.monitor_metric},
            lr_warm_up, 
        ]
        return (return_optim_list, return_lr_scheduler_list)

    def training_step(self, batch, batch_idx, dataloader_idx=None) -> torch.Tensor:
        return self.update_step(
            batch, 
            batch_idx, 
            stage='train',
            dataloader_idx=dataloader_idx,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        self.update_step(
            batch, 
            batch_idx, 
            stage='val',
            dataloader_idx=dataloader_idx,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        self.update_step(
            batch, 
            batch_idx, 
            stage='test',
            dataloader_idx=None if isinstance(batch, dict) else dataloader_idx,
        )

    def update_step(
            self, 
            batch: dict|list, 
            batch_idx = None, 
            dataloader_idx = None,
            stage: str = '', 
    ) -> torch.Tensor:
        sum_loss = torch.Tensor([0.0])
        model_outputs = self(batch)
        for task in model_outputs:
            task_outputs = model_outputs[task]
            metrics = self.task_metrics[task]
            if task == 'elm_classifier' and dataloader_idx in [None, 0]:
                labels = batch[task][1][0.5] if isinstance(batch, dict) else batch[1][0.5]
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs.reshape_as(labels),
                            target=labels.type_as(task_outputs),
                        )
                        sum_loss = sum_loss + metric_value if sum_loss else metric_value
                        if self.elm_mean_loss_factor:
                            mean_loss = self.elm_mean_loss_factor * task_outputs.mean().pow(2).sqrt() / task_outputs.std()
                            sum_loss = sum_loss + mean_loss
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            y_pred=(task_outputs.detach().cpu() >= 0.0).type(torch.int), 
                            y_true=labels.detach().cpu(),
                            zero_division=0,
                        )
                        if self.current_epoch<10:
                            metric_value /= 10
                    elif 'stat' in metric_name:
                        metric_value = metric_function(task_outputs).item()
                    self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True, add_dataloader_idx=False)
            elif task == 'conf_classifier' and dataloader_idx in [None, 1]:
                labels = batch[task][1] if isinstance(batch, dict) else batch[1]
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs,
                            target=labels.flatten(),
                        )
                        sum_loss = sum_loss + metric_value if sum_loss else metric_value
                        if self.conf_mean_loss_factor:
                            mean_loss = self.conf_mean_loss_factor * task_outputs.mean().pow(2).sqrt() / task_outputs.std()
                            sum_loss = sum_loss + mean_loss
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            y_pred=(task_outputs > 0.0).type(torch.int).detach().cpu(), 
                            y_true=torch.nn.functional.one_hot(
                                labels.flatten().detach().cpu(),
                                num_classes=4,
                            ),
                            zero_division=0,
                            average='macro',
                        )
                        if self.current_epoch<10:
                            metric_value /= 10
                    elif 'stat' in metric_name:
                        metric_value = metric_function(task_outputs).item()
                    self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True, add_dataloader_idx=False)
        return sum_loss

    def forward(
            self, 
            batch: dict|list, 
    ) -> dict[str,torch.Tensor]:
        results = {}
        for task in self.task_models:
            if isinstance(batch, dict):
                # dict batches for training
                results[task] = self.task_models[task](self.feature_model(batch[task][0]))
            else:
                # list batches for val/test/predict
                results[task] = self.task_models[task](self.feature_model(batch[0]))
        return results

    def on_fit_start(self):
        self.t_fit_start = time.time()
        self.zprint(f"**** Fit start with global step {self.trainer.global_step} ****")

    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        self.zprint(f"Fit time: {delt/60:0.1f} min")

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step
        # dl: torch.utils.data.DataLoader = None
        # if isinstance(self.trainer.train_dataloader, dict):
        #     dl = list(self.trainer.train_dataloader.values())[0]
        # else:
        #     dl = self.trainer.train_dataloader
        # self.log('batch_size', dl.batch_size, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.t_train_epoch_start
        global_time = time.time() - self.t_fit_start
        epoch_steps = self.global_step-self.s_train_epoch_start
        if self.is_global_zero and self.global_step > 0:
            line =  f"Ep {self.current_epoch:03d}  "
            line += f"ep/gl st: {epoch_steps:,d}/{self.global_step:,d}  "
            line += f"ep/gl min: {epoch_time/60:.2f}/{global_time/60:.2f}  " 
            for task in self.task_models:
                train_score = self.trainer.logged_metrics[f'{task}/f1_score/train']
                val_score = self.trainer.logged_metrics[f'{task}/f1_score/val']
                line += f"{task[:7]} t/v score: {train_score:.3f}/{val_score:.3f}  "
            print(line)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    def setup(self, stage=None):  # fit, validate, test, or predict
        assert self.is_global_zero == self.trainer.is_global_zero
        if self.is_global_zero:
            assert self.global_rank == 0

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        if self.trainer.world_size > 1:
            print(f"Rank {self.trainer.global_rank}: {text}")
        else:
            print(text)

@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    elm_data_file: str|Path|Any = None
    confinement_data_file: str|Path|Any = None
    elm_classifier: bool = True
    conf_classifier: bool = False
    max_elms: int|Any = None
    batch_size: int = 256
    stride_factor: int = 8
    num_workers: int|Any = None
    outlier_value: float = 6
    normalized_signal_outlier_value: float = 8
    fraction_validation: float = 0.12
    fraction_test: float = 0.0
    use_random_data: bool = False
    seed: int = None  # seed for ELM index shuffling; must be same across processes
    time_to_elm_quantile_min: float|Any = None
    time_to_elm_quantile_max: float|Any = None
    contrastive_learning: bool = False
    min_pre_elm_time: float|Any = None
    epochs_per_batch_size_reduction: int = None
    max_pow2_batch_size_reduction: int = 2
    fir_taps: int = 501  # Number of taps in the filter
    fir_bp_low: float|Any = None  # bandpass filter cut-on freq in kHz
    fir_bp_high: float|Any = None  # bandpass filter cut-off freq in kHz
    bad_shots: list = None
    num_classes: int = 4
    metadata_bounds = {
        'r_avg': None,
        'z_avg': None,
        'delz_avg': None
    }
    force_validation_shots: list = None
    force_test_shots: list = None
    max_shots_per_class: int = None
    test_only: bool = False
    n_rows: int = 8
    n_cols: int = 8
    mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
    max_confinement_event_length: int = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()

        if self.is_global_zero:
            print_fields(self)

        self.trainer: Trainer|Any = None
        self.a_coeffs = self.b_coeffs = None

        self.tasks = []
        self.state_items = []

        self.elm_raw_signal_mean: float|Any = None
        self.elm_raw_signal_stdev: float|Any = None
        if self.elm_classifier:
            self.elm_data_file = Path(self.elm_data_file).absolute()
            assert self.elm_data_file.exists()
            self.tasks.append('elm_classifier')

            self.global_shot_split: dict[str,np.ndarray] = {}
            self.global_elm_split: dict[str,Sequence] = {}
            self.time_to_elm_quantiles: dict[float,float] = {}
            self.state_items.extend([
                'global_shot_split',
                'global_elm_split',
                'elm_raw_signal_mean',
                'elm_raw_signal_stdev',
                'time_to_elm_quantiles',
            ])
            self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}

        if self.conf_classifier:
            self.confinement_data_file = Path(self.confinement_data_file).absolute()
            assert self.confinement_data_file.exists()
            self.tasks.append('conf_classifier')

            self.global_confinement_shot_split: dict[str,Sequence] = {}
            self.confinement_raw_signal_mean: float|Any = None
            self.confinement_raw_signal_stdev: float|Any = None
            self.state_items.extend([
                'global_confinement_shot_split',
                'confinement_raw_signal_mean',
                'confinement_raw_signal_stdev',
            ])
            self.confinement_datasets: dict[str,torch.utils.data.Dataset] = {}
            self.global_stage_to_events: dict = {}

        for item in self.state_items:
            assert hasattr(self, item)

    def prepare_data(self):
        # called for rank 0 only!
        self.zprint(f"**** Prepare data (rank 0 only)")
        if self.seed is None:
            self.seed = np.random.default_rng().integers(0, 2**32-1)
        self.rng = np.random.default_rng(self.seed)
        if self.elm_classifier and not self.global_shot_split and not self.global_elm_split:
            self._prepare_elm_data()
        if self.conf_classifier and not self.global_confinement_shot_split:
            self._prepare_confinement_data()

    def _prepare_elm_data(self):
        self.zprint("  Global ELM data split")
        with h5py.File(self.elm_data_file, 'r') as root:
            datafile_shots = set([int(shot_key) for shot_key in root['shots']])
            self.zprint(f"    Shots in data file: {len(datafile_shots):,d}")
            datafile_shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(datafile_shots ^ datafile_shots_from_elms) == 0
            datafile_shots = list(datafile_shots)
            datafile_elms = [int(elm_key) for elm_key in root['elms']]
            self.zprint(f"    ELMs in data file: {len(datafile_elms):,d}")
            # limit max ELMs
            if self.max_elms and len(datafile_elms) > self.max_elms:
                self.rng.shuffle(datafile_elms)
                datafile_elms = datafile_elms[:self.max_elms]
                datafile_shots = [int(root['elms'][f"{elm_index:06d}"].attrs['shot']) for elm_index in datafile_elms]
                datafile_shots = list(set(datafile_shots))
                self.zprint(f"    ELMs/shots for analysis: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            # shuffle shots in dataset
            self.zprint(f"    Shuffling global shots")
            self.rng.shuffle(datafile_shots)
            # global shot split
            self.zprint("    Global shot split")
            n_test_shots = int(self.fraction_test * len(datafile_shots))
            n_validation_shots = int(self.fraction_validation * len(datafile_shots))
            self.global_shot_split['test'], self.global_shot_split['validation'], self.global_shot_split['train'] = \
                np.split(datafile_shots, [n_test_shots, n_test_shots+n_validation_shots])
            for stage in ['train','validation','test']:
                self.zprint(f"      {stage.upper()} shot count: {self.global_shot_split[stage].size} ({self.global_shot_split[stage].size/len(datafile_shots)*1e2:.1f}%)")
            # global ELM split
            self.zprint("    Global ELM split")
            for stage in ['train','validation','test']:
                self.global_elm_split[stage] = [
                    i_elm for i_elm in datafile_elms
                    if root['elms'][f"{i_elm:06d}"].attrs['shot'] in self.global_shot_split[stage]
                ]
                self.zprint(f"      {stage.upper()} ELM count: {len(self.global_elm_split[stage]):,d} ({len(self.global_elm_split[stage])/len(datafile_elms)*1e2:.1f}%)")

    def _prepare_confinement_data(self):
        self.zprint("  Global confinement data split")
        if self.bad_shots is None:
            self.bad_shots = []  # Initialize to empty list if None
        check_bounds = lambda value, bounds: bounds[0] <= value <= bounds[1] if bounds else True
        global_shot_to_events: dict[int,list] = {}
        global_class_to_shots: list[list] = [[] for _ in range(self.num_classes)]
        global_class_duration: list[int] = [0] * self.num_classes
        global_class_to_events: list[int] = [0] * self.num_classes
        r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        missing_inboard = bad_inboard = 0
        with h5py.File(self.confinement_data_file) as root:
            for shot_key in root:
                if int(shot_key) in self.bad_shots:
                    continue
                inboard_order = root[shot_key].attrs.get("inboard_column_channel_order", None)
                if inboard_order is None or len(inboard_order)==0:
                    missing_inboard += 1
                    continue
                if not np.array_equal(inboard_order, np.arange(8, dtype=int)*8+1):
                    bad_inboard += 1
                    continue
                metadata = {
                    'r_avg': root[shot_key].attrs.get('r_avg'),
                    'z_avg': root[shot_key].attrs.get('z_avg'),
                    'delz_avg': root[shot_key].attrs.get('delz_avg')
                }
                if not all(
                    check_bounds(metadata[key], self.metadata_bounds[key]) 
                    for key in ['r_avg', 'z_avg', 'delz_avg'] 
                    if key in self.metadata_bounds
                ):
                    if metadata['r_avg'] is None or not check_bounds(metadata['r_avg'], self.metadata_bounds['r_avg']):
                        r_avg_exclusions += 1
                    if metadata['z_avg'] is None or not check_bounds(metadata['z_avg'], self.metadata_bounds['z_avg']):
                        z_avg_exclusions += 1
                    if metadata['delz_avg'] is None or not check_bounds(metadata['delz_avg'], self.metadata_bounds['delz_avg']):
                        delz_avg_exclusions += 1
                    continue
                # loop over events in shot
                events: list[dict] = []
                for event_key in root[shot_key]:
                    event = root[shot_key][event_key]
                    if 'labels' not in event:
                        continue
                    assert event['labels'].size == event['signals'].shape[1]
                    assert event['labels'][0] == event['labels'][-1]
                    assert event['signals'].shape[0] == 64
                    assert event['labels'][0] < self.num_classes
                    label = event['labels'][0].item()
                    duration: int = event['labels'].size
                    if duration < self.signal_window_size:
                        continue
                    if self.max_confinement_event_length:
                        duration = min(duration, self.max_confinement_event_length)
                    valid_t0 = np.zeros(duration, dtype=int)
                    valid_t0[self.signal_window_size-1::self.signal_window_size//8] = 1
                    valid_t0_indices = np.arange(valid_t0.size, dtype=int)
                    valid_t0_indices = valid_t0_indices[valid_t0 == 1]
                    n_signal_windows = len(valid_t0_indices)
                    events.append({
                        'shot': int(shot_key),
                        'event': int(event_key),
                        'shot_event_key': f"{shot_key}/{event_key}",
                        'label': label,
                        'duration': duration,
                        'sw_count': n_signal_windows,
                    })
                    global_class_to_shots[label].append(int(shot_key))
                    global_class_duration[label] += duration
                    global_class_to_events[label] += 1
                if not events:
                    continue
                global_shot_to_events[int(shot_key)] = events
        global_class_to_shots = [list(set(item)) for item in global_class_to_shots]
        # BES location exclusions
        self.zprint("    Confinement data exclusions")
        self.zprint(f"      missing inboard shot exclusions: {missing_inboard}")
        self.zprint(f"      bad inboard shot exclusions: {bad_inboard}")
        self.zprint(f"      r_avg shot exclusions: {r_avg_exclusions}")
        self.zprint(f"      z_avg shot exclusions: {z_avg_exclusions}")
        self.zprint(f"      delz_avg shot exclusions: {delz_avg_exclusions}")
        # data read
        self.zprint("    Data file summary (after shot exclusions)")
        self.zprint(f"      Shots: {len(global_shot_to_events)}")
        for i in range(self.num_classes):
            self.zprint(f"        Class {i}:  shots {len(global_class_to_shots[i])}  events {global_class_to_events[i]}  duration {global_class_duration[i]:,d}")
            assert global_class_duration[i]
        # Forced shots
        forced_test_shots_data = {}
        forced_validation_shots_data = {}
        if self.force_test_shots:
            for shot_number in self.force_test_shots:
                forced_test_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        if self.force_validation_shots:
            for shot_number in self.force_validation_shots:
                forced_validation_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        if self.max_shots_per_class is not None:
            for i_class in range(len(global_class_to_shots)-1, -1, -1):
                class_shots = global_class_to_shots[i_class]
                if len(class_shots) > self.max_shots_per_class:
                    # down-select shots for each class
                    global_class_to_shots[i_class] = self.rng.choice(
                        a=class_shots, 
                        size=self.max_shots_per_class, 
                        replace=False,
                    ).tolist()
            global_shot_to_events = {shot: global_shot_to_events[shot] for shots in global_class_to_shots for shot in shots}

        # split global data into train, val, test
        shot_numbers = list(global_shot_to_events.keys())
        self.global_confinement_shot_split = {st:[] for st in ['train','validation','test']}
        good_split = False
        while good_split == False:
            self.rng.shuffle(shot_numbers)
            good_split = True
            self.global_confinement_shot_split['train'], _test_val_shots = train_test_split(
                shot_numbers, 
                test_size=self.fraction_test + self.fraction_validation, 
                random_state=self.rng.integers(0, 2**32-1),
            )
            if self.fraction_test:
                self.global_confinement_shot_split['test'], self.global_confinement_shot_split['validation'] = train_test_split(
                    _test_val_shots,
                    test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
                    random_state=self.rng.integers(0, 2**32-1),
                )
            else:
                self.global_confinement_shot_split['validation'] = _test_val_shots
                self.global_confinement_shot_split['test'] = []

            if forced_test_shots_data or forced_validation_shots_data:
                global_shot_to_events.update(forced_validation_shots_data)
                global_shot_to_events.update(forced_test_shots_data)
                self.global_confinement_shot_split['validation'].extend(list(forced_validation_shots_data.keys()))
                self.global_confinement_shot_split['test'].extend(list(forced_test_shots_data.keys()))

            self.global_confinement_shot_split['predict'] = self.global_confinement_shot_split['test']

            # map events to stages
            self.zprint("    Final data for computation")
            for st in self.global_confinement_shot_split:
                self.global_stage_to_events[st] = [event for shot in self.global_confinement_shot_split[st] for event in global_shot_to_events[shot]]
                if len(self.global_confinement_shot_split[st]) == 0:
                    self.zprint(f"      {st.capitalize()} data: 0 shots and 0 events")
                    continue
                class_to_shots = [[] for _ in range(self.num_classes)]
                class_to_events = [0] * self.num_classes
                class_to_duration = [0] * self.num_classes
                for event in self.global_stage_to_events[st]:
                    class_to_shots[event['label']].append(event['shot'])
                    class_to_events[event['label']] += 1
                    class_to_duration[event['label']] += event['duration']
                class_to_shots = [list(set(l)) for l in class_to_shots]
                if (0 in class_to_events) or good_split == False:
                    good_split = False
                    self.zprint("    Bad split, re-running")
                    continue
                self.zprint(f"      {st.capitalize()} data: {len(self.global_confinement_shot_split[st])} shots and {len(self.global_stage_to_events[st])} events")
                for i in range(self.num_classes):
                    self.zprint(f"        Class {i}: {len(class_to_shots[i])} shots, {class_to_events[i]} events, {class_to_duration[i]:,d} timepoints")
                    assert len(class_to_shots[i]) > 0

    def setup(self, stage: str):
        # called on all ranks after "prepare_data()"
        t_tmp = time.time()
        self.zprint(f"**** Setup: {stage.upper()} (all ranks)")

        assert stage in ['fit', 'test', 'predict']
        assert self.is_global_zero == self.trainer.is_global_zero

        assert self.batch_size % self.trainer.world_size == 0
        self.zprint(f"  Global batch size: {self.batch_size}")
        self.zprint(f"  Rank batch size: {self.batch_size // self.trainer.world_size}")
        if self.outlier_value:
            self.zprint(f"  Removing raw data outliers with max(abs(signal windows)) > {self.outlier_value:.3f} V")

        if self.num_workers is None:
            self.num_workers = 8 if self.trainer.world_size>1 else 0

        if self.fir_bp_low is None and self.fir_bp_high is None:
            self.zprint("  Using raw BES signals with no FIR filter")
        else:
            self.zprint(f"  FIR filter with f_low-f_high: {self.fir_bp_low} - {self.fir_bp_high} kHz")
            if self.fir_bp_low and self.fir_bp_high:
                pass_zero = 'bandpass'
                cutoff = [self.fir_bp_low, self.fir_bp_high]
            elif self.fir_bp_low:
                pass_zero = 'highpass'
                cutoff = self.fir_bp_low
            elif self.fir_bp_high:
                pass_zero = 'lowpass'
                cutoff = self.fir_bp_high
            self.b_coeffs = scipy.signal.firwin(
                numtaps=self.fir_taps,  # must be odd
                cutoff=cutoff,  # transition width in kHz
                pass_zero=pass_zero,
                fs=1e3,  # f_sample in kHz
            )
            self.a_coeffs = np.zeros_like(self.b_coeffs)
            self.a_coeffs[0] = 1

        stages = ['train', 'validation'] if stage == 'fit' else [stage]
        self.zprint(f"  Data setup for stages: {stages}")

        if self.elm_classifier:
            t_tmp = time.time()
            self.zprint("**** ELM data setup")
            for st in stages:
                self._setup_elm_data_for_stage(st)
            self.zprint(f"  ELM data setup time: {time.time()-t_tmp:0.1f} s")
        self.barrier()

        if self.conf_classifier:
            t_tmp = time.time()
            self.zprint("**** Confinement data setup")
            for st in stages:
                self._setup_confinement_data(st)
            self.zprint(f"  Confinement data setup time: {time.time()-t_tmp:.1f} s")
        self.barrier()

    def _setup_elm_data_for_stage(self, stage: str):
        # ELM/shot splits from "prepare_data()"
        self.global_elm_split = self.broadcast(self.global_elm_split)
        self.global_shot_split = self.broadcast(self.global_shot_split)
        self.zprint(f"  {stage.upper()}")
        if stage in self.elm_datasets and isinstance(self.elm_datasets[stage], torch.utils.data.Dataset):
            self.zprint(f"    Using existing dataset")
            return
        stage_sw_metadata: list = []
        if self.is_global_zero:
            elm_indices = self.global_elm_split[stage]
            self.zprint(f"    ELM count: {len(elm_indices)}")
            outliers = 0
            sw_count = 0
            skipped_short_pre_elm_time = 0
            n_bins = 200
            signal_min = np.array(np.inf)
            signal_max = np.array(-np.inf)
            cummulative_hist = np.zeros(n_bins, dtype=int)
            last_stat_elm_index = -1
            stat_interval = 500
            stat_count = 0
            t_stat = time.time()
            with h5py.File(self.elm_data_file, 'r') as h5_file:
                elms: h5py.Group = h5_file['elms']
                for i_elm, elm_index in enumerate(elm_indices):
                    if i_elm%(len(elm_indices)//10) == 0:
                        self.zprint(f"    Reading ELM event {i_elm:04d}/{len(elm_indices):04d}")
                    elm_event: h5py.Group = elms[f"{elm_index:06d}"]
                    shot = int(elm_event.attrs['shot'])
                    assert elm_event["bes_signals"].shape[0] == 64
                    assert elm_event['bes_time'].size == elm_event["bes_signals"].shape[1]
                    bes_time = np.array(elm_event['bes_time'], dtype=np.float32)
                    t_start: float = elm_event.attrs['t_start']
                    t_stop: float = elm_event.attrs['t_stop'] - 0.05
                    if self.min_pre_elm_time and (t_stop-t_start) < self.min_pre_elm_time:
                        skipped_short_pre_elm_time += 1
                        continue
                    i_start: int = np.flatnonzero(bes_time >= t_start)[0]
                    i_stop: int = np.flatnonzero(bes_time <= t_stop)[-1]
                    i_window_stop = i_stop
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    assert signals.shape[0] == 64
                    assert signals.shape[1] == bes_time.size
                    while True:
                        i_window_start = i_window_stop - self.signal_window_size
                        if i_window_start < i_start:
                            break  # break while loop
                        # remove outliers in raw signals
                        if self.outlier_value:
                            signal_window = signals[..., i_window_start:i_window_stop]
                            if np.abs(signal_window).max() > self.outlier_value:
                                i_window_stop -= self.signal_window_size // self.stride_factor
                                outliers += 1
                                continue
                        sw_count += 1
                        stage_sw_metadata.append({
                            'elm_index': elm_index,
                            'shot': shot,
                            'i_t0': i_window_start,
                            'time_to_elm': bes_time[i_stop] - bes_time[i_window_stop]
                        })
                        if sw_count % stat_interval == 0:
                            stat_count += 1
                            if elm_index != last_stat_elm_index:
                                # apply FIR (if used) for stats calculations
                                if self.b_coeffs is not None:
                                    fsignals = np.array(
                                        scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
                                        dtype=np.float32,
                                    )
                                else:
                                    fsignals = signals
                            last_stat_elm_index = elm_index
                            fsignal_window = fsignals[..., i_window_start:i_window_stop]
                            signal_min = np.min([signal_min, fsignal_window.min()])
                            signal_max = np.max([signal_max, fsignal_window.max()])
                            hist, bin_edges = np.histogram(
                                fsignal_window,
                                bins=n_bins,
                                range=(-10.4, 10.4),
                            )
                            cummulative_hist += hist
                        i_window_stop -= self.signal_window_size // self.stride_factor
            self.zprint(f"    Skipped ELMs for short pre-ELM time: {skipped_short_pre_elm_time}")
            self.zprint(f"    Skipped outliers: {outliers:,d}")
            self.zprint(f"    Global signal window count (unprocessed): {len(stage_sw_metadata):,d}")

            # stats
            # n_bins = 200
            # signal_min = np.array(np.inf)
            # signal_max = np.array(-np.inf)
            # cummulative_hist = np.zeros(n_bins, dtype=int)
            # last_stat_elm_index = -1
            # r0_sw_metadata = stage_sw_metadata[:len(stage_sw_metadata)//self.trainer.world_size]
            # stat_interval = np.max([self.stride_factor, len(r0_sw_metadata)//int(2e3)])
            # t_stat = time.time()
            # stat_count = 0
            # with h5py.File(self.elm_data_file) as root:
            #     for sw in r0_sw_metadata[::stat_interval]:
            #         stat_count += 1
            #         elm_index = sw['elm_index']
            #         if elm_index != last_stat_elm_index:
            #             elm_event: h5py.Group = root['elms'][f'{elm_index:06d}']
            #             signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
            #             # apply FIR (if used) for stats calculations
            #             if self.b_coeffs is not None:
            #                 signals = np.array(
            #                     scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
            #                     dtype=np.float32,
            #                 )
            #             signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
            #         last_stat_elm_index = elm_index
            #         i_t0 = sw['i_t0']
            #         signal_window = signals[i_t0: i_t0 + self.signal_window_size, :, :]
            #         assert signal_window.shape[0] == self.signal_window_size
            #         signal_min = np.min([signal_min, signal_window.min()])
            #         signal_max = np.max([signal_max, signal_window.max()])
            #         hist, bin_edges = np.histogram(
            #             signal_window,
            #             bins=n_bins,
            #             range=(-10.4, 10.4),
            #         )
            #         cummulative_hist += hist
            bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
            mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
            stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
            exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
            self.zprint(f"    Signal stats (post-FIR, if used):  mean {mean:.3f}  stdev {stdev:.3f}  exkurt {exkurt:.3f}  min/max {signal_min:.3f}/{signal_max:.3f}")
            if stage == 'train' and not self.elm_raw_signal_mean:
                self.zprint(f"    Using {stage.upper()} for standardizing mean and stdev")
                self.elm_raw_signal_mean = mean.item()
                self.elm_raw_signal_stdev = stdev.item()
                self.save_hyperparameters({
                    'raw_signal_mean': self.elm_raw_signal_mean,
                    'raw_signal_stdev': self.elm_raw_signal_stdev,
                })
            self.zprint(f"    Stat time {time.time()-t_stat:.1f} s  ({stat_count:,d} samples)")

        self.elm_raw_signal_mean = self.broadcast(self.elm_raw_signal_mean)
        self.elm_raw_signal_stdev = self.broadcast(self.elm_raw_signal_stdev)
        self.zprint(f"    Standarizing signals with mean {self.elm_raw_signal_mean:.3f} and std {self.elm_raw_signal_stdev:.3f}")

        # time-to-ELM quantiles
        if self.is_global_zero:
            if stage == 'train' and not self.time_to_elm_quantiles:
                quantiles = [0.5]
                time_to_elm_labels = [sig_win['time_to_elm'] for sig_win in stage_sw_metadata]
                quantile_values = np.quantile(time_to_elm_labels, quantiles)
                self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
                self.save_hyperparameters({
                    'time_to_elm_quantiles': self.time_to_elm_quantiles,
                })
                self.zprint(f"    Time-to-ELM quantiles for binary labels:")
                for q, qval in self.time_to_elm_quantiles.items():
                    self.zprint(f"      Quantile {q:.2f}: {qval:.1f} ms")
        self.time_to_elm_quantiles = self.broadcast(self.time_to_elm_quantiles)

        # restrict data according to quantiles
        if self.is_global_zero:
            if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
                time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in stage_sw_metadata])
                time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
                if self.contrastive_learning:
                    self.zprint(f"    Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                    for i in np.arange(len(stage_sw_metadata)-1, -1, -1, dtype=int):
                        if (stage_sw_metadata[i]['time_to_elm'] > time_to_elm_min) and \
                            (stage_sw_metadata[i]['time_to_elm'] < time_to_elm_max):
                            stage_sw_metadata.pop(i)
                else:
                    self.zprint(f"    Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                    for i in np.arange(len(stage_sw_metadata)-1, -1, -1, dtype=int):
                        if (stage_sw_metadata[i]['time_to_elm'] < time_to_elm_min) or \
                            (stage_sw_metadata[i]['time_to_elm'] > time_to_elm_max):
                            stage_sw_metadata.pop(i)
            remainder = len(stage_sw_metadata) % self.trainer.world_size
            if remainder:
                stage_sw_metadata = stage_sw_metadata[:-remainder]
            assert len(stage_sw_metadata) % self.trainer.world_size == 0
            self.zprint(f"    Global signal window count (final): {len(stage_sw_metadata):,d}")
            self.zprint(f"    Batches per epoch: {len(stage_sw_metadata)/self.batch_size:,.1f}")
        stage_sw_metadata = self.broadcast(stage_sw_metadata)

        # split signal windows across ranks
        rankwise_sw_split = None
        if self.is_global_zero:
            rankwise_sw_split = np.array_split(stage_sw_metadata, self.trainer.world_size)
        rankwise_sw_split = self.broadcast(rankwise_sw_split)

        # get rank-wise ELM signals
        sw_for_rank = list(rankwise_sw_split[self.trainer.global_rank])
        elms_for_rank = np.unique(np.array([item['elm_index'] for item in sw_for_rank],dtype=int))
        elm_to_signals = {}
        with h5py.File(self.elm_data_file) as root:
            for elm_index in elms_for_rank:
                elm_group: h5py.Group = root['elms'][f"{elm_index:06d}"]
                signals = np.array(elm_group["bes_signals"], dtype=np.float32)  # (64, <time>)
                if self.b_coeffs is not None:
                    signals = np.array(
                        scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
                        dtype=np.float32,
                    )
                signals = np.transpose(signals).reshape(1, -1, 8, 8)  # reshape to (time, pol, rad)
                # FIR first (if used), then normalize
                elm_to_signals[elm_index] = (signals - self.elm_raw_signal_mean) / self.elm_raw_signal_stdev
        assert len(elm_to_signals) == len(elms_for_rank)
        signal_memory_size = sum([array.nbytes for array in elm_to_signals.values()])
        self.rprint(f"    Signal memory size: {signal_memory_size/(1024**3):.3f} GB")

        # rank-wise datasets
        if stage in ['train', 'validation', 'test']:
            self.elm_datasets[stage] = ELM_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                time_to_elm_quantiles=self.time_to_elm_quantiles,
                sw_list=sw_for_rank,
                signal_list=elm_to_signals,
                quantile_min=self.time_to_elm_quantile_min,
                quantile_max=self.time_to_elm_quantile_max,
                contrastive_learning=self.contrastive_learning,
            )
        elif stage == 'predict':
            pass

    def _setup_confinement_data(self, stage: str):
        t_tmp = time.time()
        self.zprint(f"  {stage.upper()}")
        self.global_stage_to_events = self.broadcast(self.global_stage_to_events)
        global_stage_events = self.global_stage_to_events[stage]
        rankwise_stage_event_split = None
        if self.is_global_zero:
            if self.trainer.world_size == 0:
                rankwise_stage_event_split = [global_stage_events,]
            else:
                # sort events by largest to smallest sig win count
                global_stage_events = sorted(
                    global_stage_events,
                    key=lambda e: e['sw_count'],
                    reverse=True,
                )
                # assign each event to the rank with the smallest sig win count
                rankwise_stage_event_split = [{'events':[], 'sw_count':0} for _ in range(self.trainer.world_size)]
                for event in global_stage_events:
                    rankwise_stage_event_split = sorted(
                        rankwise_stage_event_split,
                        key=lambda e: e['sw_count'],
                    )
                    rankwise_stage_event_split[0]['events'].append(event)
                    rankwise_stage_event_split[0]['sw_count'] += event['sw_count']
                rankwise_sw_counts = [rd['sw_count'] for rd in rankwise_stage_event_split]
                self.zprint(f"    Rank-wise signal window count (approx): {rankwise_sw_counts}")
                rankwise_stage_event_split = [re['events'] for re in rankwise_stage_event_split]
                # for i, re in enumerate(rankwise_stage_event_split):
                #     self.zprint(f"    Rank {i} shot/ev keys: {[e['shot_event_key'] for e in re[0:4]]}")
        rankwise_stage_event_split = self.broadcast(rankwise_stage_event_split)

        # package data for rank
        rankwise_events = rankwise_stage_event_split[self.trainer.global_rank]
        n_events = len(rankwise_events)
        # pre-allocate signal array
        time_count = sum([event['duration'] for event in rankwise_events])
        packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
        rankwise_events_2 = []
        start_index = 0
        outlier_count = 0
        with h5py.File(self.confinement_data_file, 'r') as root:
            for i, event_data in enumerate(rankwise_events):
                if n_events >= 10 and i % (n_events//10) == 0:
                    self.zprint(f"    Reading event {i:04d}/{n_events:04d}")
                shot = event_data['shot']
                event = event_data['event']
                label = event_data['label']
                duration = event_data['duration']
                sw_count = event_data['sw_count']
                event_group = root[str(shot)][str(event)]
                labels = np.array(event_group["labels"], dtype=int)
                assert labels[0] == label and labels[-1] == label
                signals = np.array(event_group["signals"][:, :], dtype=np.float32)
                assert signals.shape[0] == 64
                assert labels.size == signals.shape[1]
                if self.max_confinement_event_length and labels.size>self.max_confinement_event_length:
                    labels = labels[:self.max_confinement_event_length]
                    signals = signals[:,:self.max_confinement_event_length]
                assert labels.size == duration
                signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                valid_t0 = np.zeros(labels.size, dtype=int)
                valid_t0[self.signal_window_size-1::self.signal_window_size//8] = 1
                valid_t0_indices = np.arange(valid_t0.size, dtype=int)
                valid_t0_indices = valid_t0_indices[valid_t0 == 1]
                for i in valid_t0_indices:
                    assert i - self.signal_window_size + 1 >= 0  # start slice test
                    assert i+1 <= valid_t0.size  # end slice test
                assert len(valid_t0_indices) == sw_count
                # remove outliers in raw signals
                if self.outlier_value:
                    for ii in valid_t0_indices:
                        if np.max(np.abs(signals[ii-self.signal_window_size+1:ii+1, ...])) > self.outlier_value:
                            outlier_count += 1
                            valid_t0[ii] = 0
                # FIR filter, if used
                if self.b_coeffs is not None:
                    signals = np.array(
                        scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
                        dtype=np.float32,
                    )
                packaged_signals[start_index:start_index + signals.shape[0], ...] = signals
                start_index += signals.shape[0]
                event_2 = event_data.copy()
                event_2['labels'] = labels
                event_2['valid_t0'] = valid_t0
                rankwise_events_2.append(event_2)

        self.rprint(f"    Outlier count: {outlier_count}")
        assert start_index == packaged_signals.shape[0]
        packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in rankwise_events_2], axis=0)
        packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in rankwise_events_2], axis=0)
        assert packaged_labels.size == packaged_valid_t0.size
        assert packaged_labels.size == packaged_signals.shape[0]

        packaged_window_start = []
        index = 0
        for event in rankwise_events_2:
            packaged_window_start.append(index)
            index += event['labels'].size
        packaged_window_start = np.array(packaged_window_start, dtype=int)
        assert len(rankwise_events_2) == len(packaged_window_start)

        packaged_shot_event_key = np.array(
            [event['shot_event_key'] for event in rankwise_events_2],
            dtype=str,
        )
        assert len(rankwise_events_2) == len(packaged_shot_event_key)

        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        for i in packaged_valid_t0_indices:
            assert i - self.signal_window_size + 1 >= 0  # start slice
            assert i+1 <= packaged_valid_t0.size  # end slice

        # match valid t0 indices count across ranks
        if self.trainer.world_size > 1:
            np.random.default_rng().shuffle(packaged_valid_t0_indices)
            count_valid_t0_indices = len(packaged_valid_t0_indices)
            self.rprint(f"    Valid t0 indices (unmatched): {count_valid_t0_indices:,d}")
            all_rank_count_valid_indices: list = [None for _ in range(self.trainer.world_size)]
            for i in range(self.trainer.world_size):
                all_rank_count_valid_indices[i] = self.trainer.strategy.broadcast(count_valid_t0_indices, src=i)
            length_limit = min(all_rank_count_valid_indices)
            packaged_valid_t0_indices = packaged_valid_t0_indices[:length_limit]


        # stats
        if self.is_global_zero:
            signal_min = np.inf
            signal_max = -np.inf
            n_bins = 200
            cummulative_hist = np.zeros(n_bins, dtype=int)
            stat_interval = max(1, packaged_valid_t0_indices.size//int(2e3))
            t_stat = time.time()
            stat_count = 0
            for i in packaged_valid_t0_indices[::stat_interval]:
                stat_count += 1
                # signals are post-FIR, if used
                signal_window = packaged_signals[i-self.signal_window_size+1:i+1, ...]
                signal_min = min(signal_min, signal_window.min())
                signal_max = max(signal_max, signal_window.max())
                hist, bin_edges = np.histogram(
                    signal_window,
                    bins=n_bins,
                    range=[-10.4, 10.4],
                )
                cummulative_hist += hist
            bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
            mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
            stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
            exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
            self.zprint(f"    Signal stats (after FIR, if used): mean {mean:.3f} stdev {stdev:.3f} exkurt {exkurt:.3f} min/max {signal_min:.3f}/{signal_max:.3f}")
            if stage == 'train' and not self.confinement_raw_signal_mean:
                self.zprint(f"    Using {stage.upper()} for standarizing mean and stdev")
                self.confinement_raw_signal_mean = mean
                self.confinement_raw_signal_stdev = stdev
                self.save_hyperparameters({
                    'signal_mean': self.confinement_raw_signal_mean.item(),
                    'signal_stdev': self.confinement_raw_signal_stdev.item(),
                })
            self.zprint(f"    Stat time {time.time()-t_stat:.1f} s  ({stat_count:,d} samples)")

        self.confinement_raw_signal_mean = self.broadcast(self.confinement_raw_signal_mean)
        self.confinement_raw_signal_stdev = self.broadcast(self.confinement_raw_signal_stdev)
        if self.elm_raw_signal_mean and self.elm_raw_signal_stdev:
            self.zprint(f"    Using ELM data mean/stdev")
            self.confinement_raw_signal_mean = self.elm_raw_signal_mean
            self.confinement_raw_signal_stdev = self.elm_raw_signal_stdev
        self.zprint(f"    Standarizing signals with mean {self.confinement_raw_signal_mean:.3f} and std {self.confinement_raw_signal_stdev:.3f}")
        packaged_signals = (packaged_signals - self.confinement_raw_signal_mean) / self.confinement_raw_signal_stdev

        self.zprint(f"    Valid t0 indices (per rank): {len(packaged_valid_t0_indices):,d}")
        self.rprint(f"    Signal memory size: {packaged_signals.nbytes/(1024**3):.4f} GB")
        self.zprint(f"    Batches per epoch: {len(packaged_valid_t0_indices)*self.trainer.world_size/self.batch_size:.1f}")
        self.zprint(f"    {stage.upper()} data time: {time.time()-t_tmp:.1f} s")

        if stage in ['train', 'validation', 'test']:
            self.confinement_datasets[stage] = Confinement_TrainValTest_Dataset(
                    signals=packaged_signals,
                    n_rows=self.n_rows,
                    n_cols=self.n_cols,
                    labels=packaged_labels,
                    sample_indices=packaged_valid_t0_indices,
                    window_start_indices=packaged_window_start,
                    signal_window_size=self.signal_window_size,
                    shot_event_keys=packaged_shot_event_key,
                )
        elif stage == 'predict':
            pass

    def _elm_train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.elm_datasets[stage],
            num_replicas=1,
            rank=0,
            shuffle=True if stage=='train' else False,
            seed=int(np.random.default_rng().integers(0, 2**32-1)),
            drop_last=True if stage=='train' else False,
        )
        if stage == 'train' and self.epochs_per_batch_size_reduction:
            batch_size_reduction_pow2_factor = min(
                self.max_pow2_batch_size_reduction, 
                self.trainer.current_epoch//self.epochs_per_batch_size_reduction,
            ) 
            batch_size = self.batch_size // (2**batch_size_reduction_pow2_factor)
            if batch_size != self.batch_size:
                self.zprint(f"Reduced global batch size: {batch_size}")
        else:
            batch_size = self.batch_size
        return torch.utils.data.DataLoader(
            dataset=self.elm_datasets[stage],
            sampler=sampler,
            batch_size=batch_size//self.trainer.world_size,  # batch size per rank
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if stage=='train' else False,
        )

    def _conf_train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.confinement_datasets[stage],
            num_replicas=1,
            rank=0,
            shuffle=True if stage=='train' else False,
            seed=int(np.random.default_rng().integers(0, 2**32-1)),
            drop_last=True if stage=='train' else False,
        )
        if stage == 'train' and self.epochs_per_batch_size_reduction:
            batch_size_reduction_pow2_factor = min(
                self.max_pow2_batch_size_reduction, 
                self.trainer.current_epoch//self.epochs_per_batch_size_reduction,
            ) 
            batch_size = self.batch_size // (2**batch_size_reduction_pow2_factor)
            if batch_size != self.batch_size:
                self.zprint(f"Reduced global batch size: {batch_size}")
        else:
            batch_size = self.batch_size
        return torch.utils.data.DataLoader(
            dataset=self.confinement_datasets[stage],
            sampler=sampler,
            batch_size=batch_size//self.trainer.world_size,  # batch size per rank
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if stage=='train' else False,
        )

    def train_dataloader(self) -> CombinedLoader:
        loaders = {}
        if self.elm_classifier:
            loaders['elm_classifier'] = self._elm_train_val_test_dataloaders('train')
        if self.conf_classifier:
            loaders['conf_classifier'] = self._conf_train_val_test_dataloaders('train')
        combined_loader = CombinedLoader(
            iterables=loaders,
            mode='max_size_cycle',
        )
        _ = iter(combined_loader)
        return combined_loader

    def val_dataloader(self) -> CombinedLoader:
        loaders = {}
        if self.elm_classifier:
            loaders['elm_classifier'] = self._elm_train_val_test_dataloaders('validation')
        if self.conf_classifier:
            loaders['conf_classifier'] = self._conf_train_val_test_dataloaders('validation')
        combined_loader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def test_dataloader(self) -> CombinedLoader:
        loaders = {}
        if self.elm_classifier:
            loaders['elm_classifier'] = self._elm_train_val_test_dataloaders('test')
        if self.conf_classifier:
            loaders['conf_classifier'] = self._conf_train_val_test_dataloaders('test')
        combined_loader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def predict_dataloader(self) -> None:
        pass

    def get_state_dict(self) -> dict:
        state_dict = {item: getattr(self, item) for item in self.state_items}
        return state_dict

    def load_state_dict(self, state: dict) -> None:
        for item in self.state_items:
            setattr(self, item, state[item])
            self.zprint(f"Loading state item {item} = {getattr(self, item)}")

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        if self.trainer.world_size > 1:
            self.barrier()
            print(f"Rank {self.trainer.global_rank}: {text}")
            self.barrier()
        else:
            print(text)

    def barrier(self):
        if self.trainer.world_size > 0:
            self.trainer.strategy.barrier()

    def broadcast(self, obj):
        if self.trainer.world_size > 0:
            obj = self.trainer.strategy.broadcast(obj)
        return obj


@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(_Base_Class, torch.utils.data.Dataset):
    signal_window_size: int = 0
    sw_list: list|Any = None # global signal window data mapping to dataset index
    signal_list: dict|Any = None # rank-wise signals (map to ELM indices)
    time_to_elm_quantiles: dict[float, float]|Any = None
    quantile_min: float|Any = None
    quantile_max: float|Any = None
    contrastive_learning: bool = False

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        for elm_index in self.signal_list:
            self.signal_list[elm_index] = torch.from_numpy(
                self.signal_list[elm_index]
            )

    def __len__(self) -> int:
        return len(self.sw_list)
    
    def __getitem__(self, i: int) -> tuple:
        sw_metadata = self.sw_list[i]
        i_t0 = sw_metadata['i_t0']
        time_to_elm = sw_metadata['time_to_elm']
        elm_index = sw_metadata['elm_index']
        signals = self.signal_list[elm_index]
        signal_window = signals[..., i_t0 : i_t0 + self.signal_window_size, :, :]
        quantile_binary_label = {q: int(time_to_elm<=qval) for q, qval in self.time_to_elm_quantiles.items()}
        return signal_window, quantile_binary_label, time_to_elm


@dataclasses.dataclass(eq=False)
class Confinement_TrainValTest_Dataset(torch.utils.data.Dataset):
    signals: np.ndarray
    n_rows: int
    n_cols: int
    labels: np.ndarray
    sample_indices: np.ndarray
    window_start_indices: np.ndarray
    signal_window_size: int
    shot_event_keys: np.ndarray

    def __post_init__(self) -> None:
        self.labels = torch.from_numpy(self.labels)
        self.signals = torch.from_numpy(np.ascontiguousarray(self.signals)[np.newaxis, ...])
        self.window_start_indices = torch.from_numpy(self.window_start_indices)
        self.sample_indices = torch.from_numpy(self.sample_indices)

        assert (
            self.signals.ndim == 4 and
            self.signals.shape[0] == 1 and
            self.signals.shape[2] == self.n_rows and
            self.signals.shape[3] == self.n_cols
        ), "Signals have incorrect shape"
        assert self.signals.shape[1] == self.labels.shape[0]
        assert torch.max(self.sample_indices) < self.labels.shape[0]

    def __len__(self) -> int:
        return self.sample_indices.numel()
    
    def __getitem__(self, i: int) -> tuple:
        # Retrieve the index from sample_indices that is guaranteed to have enough previous data
        i_t0 = self.sample_indices[i]
        # Define the start index for the signal window to look backwards
        start_index = i_t0 - self.signal_window_size + 1
        # Retrieve the signal window from start_index to i_t0 (inclusive)
        signal_window = self.signals[:, start_index:i_t0 + 1, :, :]
        # The label is typically the current index in real-time scenarios
        label = self.labels[i_t0: i_t0 + 1]
        # Look up the correct confinement_mode_key based on i_t0
        i_key = (self.window_start_indices <= i_t0).nonzero().max()
        shot_event_key = self.shot_event_keys[i_key]
        # Convert the key to an integer by removing non-numeric characters and converting to int
        confinement_mode_id_int = int(shot_event_key.replace('/', ''))
        # Convert to tensor
        confinement_mode_id_tensor = torch.tensor([confinement_mode_id_int], dtype=torch.int64)
        return signal_window, label, confinement_mode_id_tensor


def main(
        elm_data_file: str|Path = None,
        confinement_data_file: str|Path = None,
        elm_classifier=True,
        conf_classifier=False,
        max_elms: int|Any = None,
        signal_window_size = 1024,
        experiment_name = 'experiment_default',
        # model
        lr = 1e-3,
        weight_decay = 1e-4,
        lr_scheduler_patience = 20,
        lr_warmup_epochs: int = 5,
        monitor_metric = None,
        use_optimizer = 'SGD',
        elm_mean_loss_factor = None,
        conf_mean_loss_factor = None,
        initial_weight_factor = 1.0,
        # loggers
        log_freq = 100,
        use_wandb = False,
        # callbacks
        early_stopping_min_delta = 1e-3,
        early_stopping_patience = 20,
        # trainer
        max_epochs = 2,
        gradient_clip_val = None,
        gradient_clip_algorithm = None,
        skip_train: bool = False,
        precision = None,
        enable_progress_bar = False,
        # data
        batch_size = 64,
        fraction_validation = 0.15,
        fraction_test = 0.0,
        num_workers = None,
        time_to_elm_quantile_min: float|Any = None,
        time_to_elm_quantile_max: float|Any = None,
        contrastive_learning: bool = True,
        min_pre_elm_time: float|Any = None,
        fir_bp_low = None,
        fir_bp_high = None,
        epochs_per_batch_size_reduction: int = None,
        max_pow2_batch_size_reduction: int = 2,
        max_shots_per_class: int = None,
        max_confinement_event_length: int = None,
):

    # SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    global_rank = int(os.getenv("SLURM_PROCID", default=0))
    local_rank = int(os.getenv("SLURM_LOCALID", default=0))
    node_rank = int(os.getenv("SLURM_NODEID", default=0))

    is_global_zero = (global_rank == 0)

    def zprint(text):
        if is_global_zero:
            print(text)

    def rprint(text):
        if world_size==1:
            print(text)
        else:
            print(f"Global rank {global_rank}: {text}")

    ### model
    lit_model = Model(
        elm_classifier=elm_classifier,
        conf_classifier=conf_classifier,
        signal_window_size=signal_window_size,
        lr=lr,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_warmup_epochs=lr_warmup_epochs,
        weight_decay=weight_decay,
        monitor_metric=monitor_metric,
        use_optimizer=use_optimizer,
        is_global_zero=is_global_zero,
        elm_mean_loss_factor=elm_mean_loss_factor,
        conf_mean_loss_factor=conf_mean_loss_factor,
        initial_weight_factor=initial_weight_factor,
    )

    monitor_metric = lit_model.monitor_metric
    lit_model.save_hyperparameters({
        'gradient_clip_val': gradient_clip_val, 
        'gradient_clip_algorithm': gradient_clip_algorithm, 
        'precision': precision,
    })

    ### callbacks
    metric_mode = 'min' if 'loss' in monitor_metric else 'max'
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
    # if world_size == 1:
    #     callbacks.append(BatchSizeFinder(init_val=128))

    ### loggers
    loggers = []
    experiment_dir = Path(experiment_name).absolute()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
    trial_name = f"r{slurm_identifier}_{datetime_str}" if slurm_identifier else f"r{datetime_str}"
    tb_logger = TensorBoardLogger(
        save_dir=experiment_dir.parent,
        name=experiment_name,
        version=trial_name,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    trial_dir = Path(tb_logger.log_dir).absolute()
    zprint(f"Trial directory: {trial_dir}")
    if use_wandb:
        wandb.login()
        wandb_logger = WandbLogger(
            save_dir=experiment_dir,
            project=experiment_name,
            name=trial_name,
        )
        wandb_logger.watch(
            model=lit_model, 
            log='all', 
            log_freq=log_freq,
        )
        loggers.append(wandb_logger)

    zprint(f"World size {world_size} on {num_nodes} node(s)")
    rprint(f"Local rank {local_rank} on node {node_rank}")

    zprint("Model Summary:")
    zprint(ModelSummary(lit_model, max_depth=-1))

    ### initialize trainer
    if precision is None:
        precision = '16-mixed' if torch.cuda.is_available() else 32
    trainer = Trainer(
        max_epochs = max_epochs,
        gradient_clip_val = gradient_clip_val,
        gradient_clip_algorithm = gradient_clip_algorithm,
        logger = loggers,
        log_every_n_steps = log_freq,
        callbacks = callbacks,
        enable_checkpointing = True,
        enable_progress_bar = enable_progress_bar,
        enable_model_summary = False,
        precision = precision,
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            static_graph=True,
        ) if world_size>1 else 'auto',
        num_nodes = num_nodes,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=10,
    )

    assert trainer.node_rank == node_rank
    assert trainer.world_size == world_size
    assert trainer.local_rank == local_rank
    assert trainer.global_rank == global_rank
    assert trainer.is_global_zero == is_global_zero

    ### data
    lit_datamodule = Data(
        signal_window_size=signal_window_size,
        elm_data_file=elm_data_file,
        confinement_data_file=confinement_data_file,
        elm_classifier=lit_model.elm_classifier,
        conf_classifier=lit_model.conf_classifier,
        max_elms=max_elms,
        batch_size=batch_size,
        fraction_test=fraction_test,
        fraction_validation=fraction_validation,
        num_workers=num_workers,
        time_to_elm_quantile_min=time_to_elm_quantile_min,
        time_to_elm_quantile_max=time_to_elm_quantile_max,
        contrastive_learning=contrastive_learning,
        is_global_zero=is_global_zero,
        min_pre_elm_time=min_pre_elm_time,
        fir_bp_low=fir_bp_low,
        fir_bp_high=fir_bp_high,
        epochs_per_batch_size_reduction=epochs_per_batch_size_reduction,
        max_pow2_batch_size_reduction=max_pow2_batch_size_reduction,
        max_shots_per_class=max_shots_per_class,
        max_confinement_event_length=max_confinement_event_length,
    )

    if skip_train is False:
        trainer.fit(lit_model, datamodule=lit_datamodule)
        if fraction_test:
            trainer.test(lit_model, lit_datamodule)

    if use_wandb:
        wandb.finish()

if __name__=='__main__':
    main(
        elm_classifier=True,
        conf_classifier=True,
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/labeled_elm_events.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=100,
        batch_size=256,
        lr=1e-3,
        max_epochs=2,
        num_workers=0,
        log_freq=20,
        fraction_validation=0.25,
        fraction_test=0.0,
        time_to_elm_quantile_min=0.4,
        time_to_elm_quantile_max=0.6,
        contrastive_learning=True,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        # fir_bp_low=4,
        # fir_bp_high=200,
        max_shots_per_class=8,
        max_confinement_event_length=int(20e3),
        enable_progress_bar=True,
        elm_mean_loss_factor=1,
        conf_mean_loss_factor=1,
        initial_weight_factor=1,
        # use_wandb=True,
    )