from pathlib import Path
import dataclasses
from datetime import datetime
from typing import OrderedDict, Sequence
import os
import time

import numpy as np
import scipy.signal
import sklearn.metrics
from sklearn.model_selection import train_test_split 
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
    LearningRateMonitor, EarlyStopping, ModelCheckpoint
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
    signal_window_size: int = 512
    is_global_zero: bool = False
    world_size: int = None
    world_rank: int = None

    def __post_init__(self):
        assert np.log2(self.signal_window_size).is_integer(), \
            'Signal window must be power of 2'


@dataclasses.dataclass(eq=False)
class Model(LightningModule, _Base_Class):
    elm_classifier: bool = True
    conf_classifier: bool = False
    lr: float = 1e-3  # maximum LR used by first layer
    lr_scheduler_patience: int = 100
    lr_scheduler_threshold: float = 1e-3
    lr_warmup_epochs: int = 5
    # lr_layerwise_decrement: float = 1.
    weight_decay: float = 1e-6
    leaky_relu_slope: float = 2e-2
    monitor_metric: str = None
    use_optimizer: str = 'SGD'
    # elm_loss_weight: float = None
    # conf_loss_weight: float = None
    initial_weight_factor: float = 1.0
    # feature_batchnorm: bool = False
    # task_batchnorm: bool = False
    no_bias: bool = False
    batch_norm: bool = False
    feature_model_layers: Sequence[dict[str, LightningModule]] = None
    mlp_task_models: dict[str, dict[str, LightningModule]] = None

    def __post_init__(self):

        # init superclasses
        super().__init__()
        super(LightningModule, self).__post_init__()
        self.save_hyperparameters()

        if self.is_global_zero:
            print_fields(self)

        # input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.feature_model: LightningModule = None
        self.feature_space_size: int = None
        self.make_feature_model()

        # default task sub-model and metrics
        assert self.feature_space_size
        if self.mlp_task_models is None:
            self.mlp_task_models = {
                'elm_class': {  # specifications for a single MLP task
                    'layers': (self.feature_space_size, 32, 1),
                    'metrics': {
                        'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
                        'f1_score': sklearn.metrics.f1_score,
                        'precision_score': sklearn.metrics.precision_score,
                        'recall_score': sklearn.metrics.recall_score,
                        'mean_stat': torch.mean,
                        'std_stat': torch.std,
                    },
                    'monitor_metric': 'f1_score/val',
                },
            }        

        # make task sub-models
        self.task_names = list(self.mlp_task_models.keys())
        self.task_metrics: dict[str, dict] = {}
        self.task_models: torch.nn.ModuleDict = torch.nn.ModuleDict()
        for task_name, task_dict in self.mlp_task_models.items():
            self.zprint(f'Task sub-model: {task_name}')
            task_layers = task_dict['layers']
            task_metrics = task_dict['metrics']
            self.task_models[task_name] = self.make_mlp_classifier(mlp_layers=task_layers)
            self.task_metrics[task_name] = task_metrics.copy()
            if self.monitor_metric is None:
                # if not specified, use `monitor_metric` from first task
                self.monitor_metric = f"{task_name}/{task_dict['monitor_metric']}"

        # initialize model parameters
        self.total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.zprint(f"Total model parameters: {self.total_parameters:,}")

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
                            sqrt_k = np.sqrt(3*self.initial_weight_factor / n_in)
                            param.data.uniform_(-sqrt_k, sqrt_k)
                            self.zprint(f"  {name}: initialized to uniform +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")
                    else:
                        raise ValueError

                print(f"Batch evaluation (batch_size=512) with randn() data")
                example_batch_data = torch.randn(
                    size=[512]+list(self.input_data_shape[1:]),
                    dtype=torch.float32,
                )
                example_batch_output = self(example_batch_data)
                for task_name, task_output in example_batch_output.items():
                    if task_output.mean().abs() / task_output.std() > 0.25:
                        good_init = False
                        break
                if good_init == True:
                    for task_name, task_output in example_batch_output.items():
                        self.zprint(f"  Task {task_name} output shape: {task_output.shape}")
                        self.zprint(f"  Task {task_name} output mean {task_output.mean():.4f} stdev {task_output.std():.4f} min/max {task_output.min():.3f}/{task_output.max():.3f}")

        self.zprint(f"Total model parameters: {self.param_count(self):,d}")
        return

    @staticmethod
    def param_count(model: LightningModule) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def make_feature_model(self) -> None:
        self.zprint("Feature space sub-model")
        if self.feature_model_layers is None:
            self.feature_model_layers = (
                {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias':True},
                {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1, 'bias':True},
                {'out_channels': 8, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias':True},
                {'out_channels': 8, 'kernel': (1, 3, 3), 'stride': 1, 'bias':True},
                {'out_channels': 16, 'kernel': (1, 4, 4), 'stride': 1, 'bias':True},
            )

        feature_layer_dict = OrderedDict()
        data_shape = self.input_data_shape
        self.zprint(f"  Input data shape: {data_shape}  (size {np.prod(data_shape)})")
        out_channels: int = None
        for i_layer, layer in enumerate(self.feature_model_layers):
            conv_layer_name = f"L{i_layer:02d}_Conv"
            bias = layer['bias'] and not self.no_bias
            conv = torch.nn.Conv3d(
                in_channels=1 if out_channels is None else out_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
                bias=bias,
            )
            n_params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            self.zprint(f"  {conv_layer_name} kern {conv.kernel_size}  stride {conv.stride}  bias {bias}  out_ch {conv.out_channels}  param {n_params:,d}  output {data_shape} (size {np.prod(data_shape)})")
            out_channels = conv.out_channels
            if i_layer > 0:
                feature_layer_dict[f"L{i_layer:02d}_Dropout"] = torch.nn.Dropout3d(0.05)
            feature_layer_dict[conv_layer_name] = conv
            feature_layer_dict[f"L{i_layer:02d}_LeRu"] = torch.nn.LeakyReLU(self.leaky_relu_slope)
            if self.batch_norm:
                feature_layer_dict[f"L{i_layer:02d}_BatchNorm"] = torch.nn.BatchNorm3d(out_channels)

        feature_layer_dict['Flatten'] = torch.nn.Flatten()
        self.feature_model = torch.nn.Sequential(feature_layer_dict)
        self.feature_space_size = self.feature_model(torch.zeros(self.input_data_shape)).numel()

        self.zprint(f"  Feature sub-model parameters: {self.param_count(self.feature_model):,d}")
        self.zprint(f"  Feature space size: {self.feature_space_size}")

    def make_mlp_classifier(
            self,
            mlp_layers: Sequence[int] = None,
    ) -> torch.nn.Module:
        self.zprint("MLP classifier sub-model")
        mlp_layer_dict = OrderedDict()
        assert mlp_layers
        n_layers = len(mlp_layers)

        for i_layer in range(n_layers-1):
            mlp_layer_name = f"L{i_layer:02d}_FC"
            bias = (True and not self.no_bias) if i_layer+1<n_layers-1 else False
            mlp_layer = torch.nn.Linear(
                in_features=mlp_layers[i_layer],
                out_features=mlp_layers[i_layer+1],
                bias=bias,
            )
            n_params = sum(p.numel() for p in mlp_layer.parameters() if p.requires_grad)
            self.zprint(f"  {mlp_layer_name}  bias {bias}  in_features {mlp_layer.in_features}  out_features {mlp_layer.out_features}  parameters {n_params:,d}")
            mlp_layer_dict[mlp_layer_name] = mlp_layer
            if i_layer+1 < n_layers-1:
                mlp_layer_dict[f"L{i_layer:02d}_LeRu"] = torch.nn.LeakyReLU(self.leaky_relu_slope)

        mlp_classifier = torch.nn.Sequential(mlp_layer_dict)

        self.zprint(f"  MLP sub-model parameters: {self.param_count(mlp_classifier):,d}")

        return mlp_classifier

    def configure_optimizers(self):
        # parameter_group = []
        # lr = self.lr
        # self.zprint("Initial layer-wise learning rates")
        # for layer_name, layer in self.feature_model.named_children():
        #     if 'bn' in layer_name:
        #         for param_name, param in layer.named_parameters():
        #             parameter_group.append({
        #                 'params': param,
        #                 'lr': self.lr,
        #             })
        #     else:
        #         for param_name, param in layer.named_parameters():
        #             assert param_name.endswith('weight') or param_name.endswith('bias')
        #             param_lr = lr if param_name.endswith('weight') else lr
        #             parameter_group.append({
        #                 'params': param,
        #                 'lr': param_lr,
        #             })
        #             self.zprint(f"  {layer_name} {param_name} {param_lr:.3e}")
        #         lr /= self.lr_layerwise_decrement
        # lr_after_feature_model = lr
        # for task_name, task_model in self.task_models.items():
        #     lr = lr_after_feature_model
        #     for layer_name, layer in task_model.named_children():
        #         if 'bn' in layer_name:
        #             for param_name, param in layer.named_parameters():
        #                 parameter_group.append({
        #                     'params': param,
        #                     'lr': self.lr,
        #                 })
        #         else:
        #             for param_name, param in layer.named_parameters():
        #                 assert param_name.endswith('weight') or param_name.endswith('bias')
        #                 param_lr = lr if param_name.endswith('weight') else lr
        #                 parameter_group.append({
        #                     'params': param,
        #                     'lr': param_lr,
        #                 })
        #                 self.zprint(f"  {task_name} {layer_name} {param_name} {param_lr:.3e}")
        #             lr /= self.lr_layerwise_decrement

        self.zprint(f"Using {self.use_optimizer.upper()} optimizer")
        optimizer = None
        optim_kwargs = {
            'params': self.parameters(),
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
        if self.use_optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                momentum=0.2, 
                **optim_kwargs,
            )
        elif self.use_optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(**optim_kwargs)
        else:
            raise ValueError

        lr_reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            min_lr=1e-4,
            # verbose=True,
        )
        lr_warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.05,
            total_iters=self.lr_warmup_epochs,
            # verbose=True,
        )
        return_optim_list = [optimizer]
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
        signal_window, time_to_elm, quantiles = batch
        model_outputs = self(signal_window)
        for task in model_outputs:
            task_outputs: torch.Tensor = model_outputs[task]
            metrics = self.task_metrics[task]
            if task == 'elm_classifier' and dataloader_idx in [None, 0]:
                labels: torch.Tensor = quantiles[0.5]
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs.reshape_as(labels),
                            target=labels.type_as(task_outputs),
                        )
                        sum_loss = sum_loss + metric_value if sum_loss else metric_value
                        if self.elm_loss_weight:
                            mean_loss = self.elm_loss_weight * task_outputs.mean().pow(2).sqrt() / task_outputs.std()
                            sum_loss = sum_loss + mean_loss
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            y_pred=(task_outputs.detach().cpu() >= 0.0).type(torch.int), 
                            y_true=labels.detach().cpu(),
                            zero_division=0,
                        )
                        # if self.current_epoch<10:
                        #     metric_value /= 10
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
                        if self.conf_loss_weight:
                            mean_loss = self.conf_loss_weight * task_outputs.mean().pow(2).sqrt() / task_outputs.std()
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
            else:
                raise ValueError

        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)
        return sum_loss

    def forward(
            self, 
            x: torch.Tensor, 
    ) -> dict[str, torch.Tensor]:
        results = {}
        for task_name, task_model in self.task_models.items():
            features = self.feature_model(x)
            results[task_name] = task_model(features)
        return results

    def on_fit_start(self):
        self.t_fit_start = time.time()
        self.zprint(f"**** Fit start with global step {self.trainer.global_step} and epoch {self.current_epoch} ****")

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step

    def on_train_batch_start(self, batch, batch_idx):
        # if batch_idx % 25 == 0:
        #     self.rprint(f"Train batch start: batch {batch_idx}")
        pass

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # self.zprint(f"Validation batch start: batch {batch_idx}, dataloader {dataloader_idx}")
        pass

    def on_train_epoch_end(self):
        if self.is_global_zero and self.global_step > 0:
            epoch_time = time.time() - self.t_train_epoch_start
            global_time = time.time() - self.t_fit_start
            epoch_steps = self.global_step-self.s_train_epoch_start
            logged_metrics = self.trainer.logged_metrics
            line =  f"Ep {self.current_epoch:03d}  "
            line += f"train/val loss {logged_metrics['sum_loss/train']:.3f}/"
            line += f"{logged_metrics['sum_loss/val']:.3f}  "
            line += f"ep/gl steps {epoch_steps:,d}/{self.global_step:,d}  "
            line += f"ep/gl time (min): {epoch_time/60:.1f}/{global_time/60:.1f}  " 
            print(line)

    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        self.rprint(f"Fit time: {delt/60:0.1f} min")

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
        if self.world_size > 1:
            print(f"  Rank {self.trainer.global_rank}: {text}")
        else:
            print(f"  {text}")


@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    elm_data_file: str|Path = None
    elm_classifier: bool = True
    confinement_data_file: str|Path = None
    conf_classifier: bool = False
    log_dir: str|Path = None
    max_elms: int = None
    batch_size: int = 256
    stride_factor: int = 8
    num_workers: int = 4
    outlier_value: float = 6
    # normalized_signal_outlier_value: float = 8
    fraction_validation: float = 0.12
    fraction_test: float = 0.0
    use_random_data: bool = False
    seed: int = None  # seed for ELM index shuffling; must be same across processes
    time_to_elm_quantile_min: float = None
    time_to_elm_quantile_max: float = None
    contrastive_learning: bool = False
    min_pre_elm_time: float = None
    epochs_per_batch_size_reduction: int = 100
    max_pow2_batch_size_reduction: int = 2
    fir_taps: int = 501  # Number of taps in the filter
    fir_bp_low: float = None  # bandpass filter cut-on freq in kHz
    fir_bp_high: float = None  # bandpass filter cut-off freq in kHz
    # bad_shots: list = None
    # num_classes: int = 4
    # metadata_bounds = {
    #     'r_avg': None,
    #     'z_avg': None,
    #     'delz_avg': None
    # }
    # force_validation_shots: list = None
    # force_test_shots: list = None
    # max_shots_per_class: int = None
    # test_only: bool = False
    # n_rows: int = 8
    # n_cols: int = 8
    # mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    # prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
    # max_confinement_event_length: int = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()

        if self.is_global_zero:
            print_fields(self)

        self.trainer: Trainer = None
        self.a_coeffs = self.b_coeffs = None

        self.elm_raw_signal_mean: float = None
        self.elm_raw_signal_stdev: float = None

        self.tasks = []
        self.state_items = []
        if self.elm_classifier:
            self.elm_data_file = Path(self.elm_data_file).absolute()
            assert self.elm_data_file.exists(), f"ELM data file {self.elm_data_file} does not exist"
            self.global_elm_split: dict[str,Sequence] = {}
            self.global_shot_split: dict[str,np.ndarray] = {}
            self.time_to_elm_quantiles: dict[float,float] = {}
            self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}
            self.tasks.append('elm_classifier')
            self._modified_batch_size_per_rank: int = None

            self.state_items.extend([
                'global_shot_split',
                'global_elm_split',
                'elm_raw_signal_mean',
                'elm_raw_signal_stdev',
                'time_to_elm_quantiles',
            ])

        if self.conf_classifier:
            self.confinement_data_file = Path(self.confinement_data_file).absolute()
            assert self.confinement_data_file.exists()
            self.tasks.append('conf_classifier')

            self.global_confinement_shot_split: dict[str,Sequence] = {}
            self.confinement_raw_signal_mean: float = None
            self.confinement_raw_signal_stdev: float = None
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
        self.zprint("\u2B1C Prepare data (rank 0 only)")
        if self.seed is None:
            self.seed = np.random.default_rng().integers(0, 2**32-1)
        self.rng = np.random.default_rng(self.seed)
        if self.elm_classifier and not self.global_shot_split and not self.global_elm_split:
            self.prepare_global_elm_data()
        # if self.conf_classifier and not self.global_confinement_shot_split:
        #     self.prepare_cglobal_onfinement_data()

    def prepare_global_elm_data(self):
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

    def prepare_global_confinement_data(self):
        # self.zprint("  Global confinement data split")
        # if self.bad_shots is None:
        #     self.bad_shots = []  # Initialize to empty list if None
        # check_bounds = lambda value, bounds: bounds[0] <= value <= bounds[1] if bounds else True
        # global_shot_to_events: dict[int,list] = {}
        # global_class_to_shots: list[list] = [[] for _ in range(self.num_classes)]
        # global_class_duration: list[int] = [0] * self.num_classes
        # global_class_to_events: list[int] = [0] * self.num_classes
        # r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        # missing_inboard = bad_inboard = 0
        # with h5py.File(self.confinement_data_file) as root:
        #     for shot_key in root:
        #         if int(shot_key) in self.bad_shots:
        #             continue
        #         inboard_order = root[shot_key].attrs.get("inboard_column_channel_order", None)
        #         if inboard_order is None or len(inboard_order)==0:
        #             missing_inboard += 1
        #             continue
        #         if not np.array_equal(inboard_order, np.arange(8, dtype=int)*8+1):
        #             bad_inboard += 1
        #             continue
        #         metadata = {
        #             'r_avg': root[shot_key].attrs.get('r_avg'),
        #             'z_avg': root[shot_key].attrs.get('z_avg'),
        #             'delz_avg': root[shot_key].attrs.get('delz_avg')
        #         }
        #         if not all(
        #             check_bounds(metadata[key], self.metadata_bounds[key]) 
        #             for key in ['r_avg', 'z_avg', 'delz_avg'] 
        #             if key in self.metadata_bounds
        #         ):
        #             if metadata['r_avg'] is None or not check_bounds(metadata['r_avg'], self.metadata_bounds['r_avg']):
        #                 r_avg_exclusions += 1
        #             if metadata['z_avg'] is None or not check_bounds(metadata['z_avg'], self.metadata_bounds['z_avg']):
        #                 z_avg_exclusions += 1
        #             if metadata['delz_avg'] is None or not check_bounds(metadata['delz_avg'], self.metadata_bounds['delz_avg']):
        #                 delz_avg_exclusions += 1
        #             continue
        #         # loop over events in shot
        #         events: list[dict] = []
        #         for event_key in root[shot_key]:
        #             event = root[shot_key][event_key]
        #             if 'labels' not in event:
        #                 continue
        #             assert event['labels'].size == event['signals'].shape[1]
        #             assert event['labels'][0] == event['labels'][-1]
        #             assert event['signals'].shape[0] == 64
        #             assert event['labels'][0] < self.num_classes
        #             label = event['labels'][0].item()
        #             duration: int = event['labels'].size
        #             if duration < self.signal_window_size:
        #                 continue
        #             if self.max_confinement_event_length:
        #                 duration = min(duration, self.max_confinement_event_length)
        #             valid_t0 = np.zeros(duration, dtype=int)
        #             valid_t0[self.signal_window_size-1::self.signal_window_size//8] = 1
        #             valid_t0_indices = np.arange(valid_t0.size, dtype=int)
        #             valid_t0_indices = valid_t0_indices[valid_t0 == 1]
        #             n_signal_windows = len(valid_t0_indices)
        #             events.append({
        #                 'shot': int(shot_key),
        #                 'event': int(event_key),
        #                 'shot_event_key': f"{shot_key}/{event_key}",
        #                 'label': label,
        #                 'duration': duration,
        #                 'sw_count': n_signal_windows,
        #             })
        #             global_class_to_shots[label].append(int(shot_key))
        #             global_class_duration[label] += duration
        #             global_class_to_events[label] += 1
        #         if not events:
        #             continue
        #         global_shot_to_events[int(shot_key)] = events
        # global_class_to_shots = [list(set(item)) for item in global_class_to_shots]
        # # BES location exclusions
        # self.zprint("    Confinement data exclusions")
        # self.zprint(f"      missing inboard shot exclusions: {missing_inboard}")
        # self.zprint(f"      bad inboard shot exclusions: {bad_inboard}")
        # self.zprint(f"      r_avg shot exclusions: {r_avg_exclusions}")
        # self.zprint(f"      z_avg shot exclusions: {z_avg_exclusions}")
        # self.zprint(f"      delz_avg shot exclusions: {delz_avg_exclusions}")
        # # data read
        # self.zprint("    Data file summary (after shot exclusions)")
        # self.zprint(f"      Shots: {len(global_shot_to_events)}")
        # for i in range(self.num_classes):
        #     self.zprint(f"        Class {i}:  shots {len(global_class_to_shots[i])}  events {global_class_to_events[i]}  duration {global_class_duration[i]:,d}")
        #     assert global_class_duration[i]
        # # Forced shots
        # forced_test_shots_data = {}
        # forced_validation_shots_data = {}
        # if self.force_test_shots:
        #     for shot_number in self.force_test_shots:
        #         forced_test_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        # if self.force_validation_shots:
        #     for shot_number in self.force_validation_shots:
        #         forced_validation_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        # if self.max_shots_per_class is not None:
        #     for i_class in range(len(global_class_to_shots)-1, -1, -1):
        #         class_shots = global_class_to_shots[i_class]
        #         if len(class_shots) > self.max_shots_per_class:
        #             # down-select shots for each class
        #             global_class_to_shots[i_class] = self.rng.choice(
        #                 a=class_shots, 
        #                 size=self.max_shots_per_class, 
        #                 replace=False,
        #             ).tolist()
        #     global_shot_to_events = {shot: global_shot_to_events[shot] for shots in global_class_to_shots for shot in shots}

        # # split global data into train, val, test
        # shot_numbers = list(global_shot_to_events.keys())
        # self.global_confinement_shot_split = {st:[] for st in ['train','validation','test']}
        # good_split = False
        # while good_split == False:
        #     self.rng.shuffle(shot_numbers)
        #     good_split = True
        #     self.global_confinement_shot_split['train'], _test_val_shots = train_test_split(
        #         shot_numbers, 
        #         test_size=self.fraction_test + self.fraction_validation, 
        #         random_state=self.rng.integers(0, 2**32-1),
        #     )
        #     if self.fraction_test:
        #         self.global_confinement_shot_split['test'], self.global_confinement_shot_split['validation'] = train_test_split(
        #             _test_val_shots,
        #             test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
        #             random_state=self.rng.integers(0, 2**32-1),
        #         )
        #     else:
        #         self.global_confinement_shot_split['validation'] = _test_val_shots
        #         self.global_confinement_shot_split['test'] = []

        #     if forced_test_shots_data or forced_validation_shots_data:
        #         global_shot_to_events.update(forced_validation_shots_data)
        #         global_shot_to_events.update(forced_test_shots_data)
        #         self.global_confinement_shot_split['validation'].extend(list(forced_validation_shots_data.keys()))
        #         self.global_confinement_shot_split['test'].extend(list(forced_test_shots_data.keys()))

        #     self.global_confinement_shot_split['predict'] = self.global_confinement_shot_split['test']

        #     # map events to stages
        #     self.zprint("    Final data for computation")
        #     for st in self.global_confinement_shot_split:
        #         self.global_stage_to_events[st] = [event for shot in self.global_confinement_shot_split[st] for event in global_shot_to_events[shot]]
        #         if len(self.global_confinement_shot_split[st]) == 0:
        #             self.zprint(f"      {st.capitalize()} data: 0 shots and 0 events")
        #             continue
        #         class_to_shots = [[] for _ in range(self.num_classes)]
        #         class_to_events = [0] * self.num_classes
        #         class_to_duration = [0] * self.num_classes
        #         for event in self.global_stage_to_events[st]:
        #             class_to_shots[event['label']].append(event['shot'])
        #             class_to_events[event['label']] += 1
        #             class_to_duration[event['label']] += event['duration']
        #         class_to_shots = [list(set(l)) for l in class_to_shots]
        #         if (0 in class_to_events) or good_split == False:
        #             good_split = False
        #             self.zprint("    Bad split, re-running")
        #             continue
        #         self.zprint(f"      {st.capitalize()} data: {len(self.global_confinement_shot_split[st])} shots and {len(self.global_stage_to_events[st])} events")
        #         for i in range(self.num_classes):
        #             self.zprint(f"        Class {i}: {len(class_to_shots[i])} shots, {class_to_events[i]} events, {class_to_duration[i]:,d} timepoints")
        #             assert len(class_to_shots[i]) > 0
        pass

    def setup(self, stage: str):
        # called on all ranks after "prepare_data()"
        self.zprint("\u2B1C " + f"Begin Data.setup(stage={stage})")

        assert stage in ['fit', 'test', 'predict'], f"Invalid stage: {stage}"
        assert self.is_global_zero == self.trainer.is_global_zero

        assert self.batch_size % self.trainer.world_size == 0, \
            f"Batch size {self.batch_size} must be divisible by world size {self.trainer.world_size}"
        # self.batch_size_per_rank = self.batch_size // self.trainer.world_size
        self.zprint(f"  Global batch size (initial): {self.batch_size}")
        self.zprint(f"  Rank batch size: {self.batch_size // self.trainer.world_size}")
        if self.outlier_value:
            self.zprint(f"  Removing raw data outliers with max(abs(signal windows)) > {self.outlier_value:.3f} V")

        if self.fir_bp_low is None and self.fir_bp_high is None:
            self.zprint("  Using raw BES signals; no FIR filter")
        else:
            self.zprint(f"  Using FIR filter with f_low-f_high: {self.fir_bp_low}-{self.fir_bp_high} kHz")
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

        self.split_global_data()
        self.save_state_dict()  # save global data split as the `state` of the data module

        sub_stages = ['train', 'validation'] if stage == 'fit' else [stage]

        if self.elm_classifier:
            t_tmp = time.time()
            self.zprint("**** ELM data setup")
            for sub_stage in sub_stages:
                signal_window_list = self.prepare_global_elm_data_for_stage(sub_stage)
            self.zprint(f"  ELM data setup time: {time.time()-t_tmp:0.1f} s")
        self.barrier()
        self.prepare_rank_data_for_stage(sub_stage, signal_window_list)

        # if self.conf_classifier:
        #     t_tmp = time.time()
        #     self.zprint("**** Confinement data setup")
        #     for sub_stage in sub_stages:
        #         self._setup_confinement_data(sub_stage)
        #     self.zprint(f"  Confinement data setup time: {time.time()-t_tmp:.1f} s")
        # self.barrier()

    def prepare_global_elm_data_for_stage(self, sub_stage: str):
        self.zprint("\u2B1C " + f"Preparing global data for stage: {sub_stage.upper()}")
        assert sub_stage in ['train', 'validation', 'test', 'predict']
        if sub_stage in self.elm_datasets and isinstance(self.elm_datasets[sub_stage], torch.utils.data.Dataset):
            self.zprint(f"  Using existing dataset")
            return
        self.global_elm_split = self.broadcast(self.global_elm_split)
        self.global_shot_split = self.broadcast(self.global_shot_split)
        global_stage_elm_indices = self.global_elm_split[sub_stage]
        assert len(global_stage_elm_indices) > 0
        global_sw_metadata_list: list = []
        if self.is_global_zero:
            self.zprint(f"    ELM count: {len(global_stage_elm_indices)}")
            sw_count = 0
            global_outliers = 0
            skipped_short_pre_elm_time = 0
            n_bins = 200
            stat_interval = 500
            cummulative_hist = np.zeros(n_bins, dtype=int)
            signal_min = np.array(np.inf)
            signal_max = np.array(-np.inf)
            last_stat_elm_index = -1
            stat_count = 0
            t_stat = time.time()
            with h5py.File(self.elm_data_file, 'r') as h5_file:
                elms: h5py.Group = h5_file['elms']
                for i_elm, elm_index in enumerate(global_stage_elm_indices):
                    if i_elm%100 == 0:
                        self.zprint(f"  Reading ELM event {i_elm:04d}/{len(global_stage_elm_indices):04d}")
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
                    if self.b_coeffs is not None:
                        signals = np.array(
                            scipy.signal.lfilter(
                                x=signals,
                                a=self.a_coeffs,
                                b=self.b_coeffs,
                            ),
                            dtype=np.float32,
                        )
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                    assert signals.shape[0] == bes_time.size
                    assert (signals.shape[1] == 8) and (signals.shape[2] == 8)
                    while True:
                        i_window_start = i_window_stop - self.signal_window_size
                        if i_window_start < i_start: break
                        # remove outliers in raw signals
                        if self.outlier_value:
                            signal_window = signals[i_window_start:i_window_stop, ...]
                            assert signal_window.shape[0] == self.signal_window_size
                            if np.abs(signal_window).max() > self.outlier_value:
                                i_window_stop -= self.signal_window_size // self.stride_factor
                                global_outliers += 1
                                continue
                        sw_count += 1
                        global_sw_metadata_list.append({
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

            self.zprint(f"  Skipped global ELMs for short pre-ELM time (threshold {self.min_pre_elm_time} ms): {skipped_short_pre_elm_time}")
            self.zprint(f"  Skipped global signal windows for outliers (threshold {self.outlier_value} V): {global_outliers}")
            self.zprint(f"  Valid global signal windows: {len(global_sw_metadata_list):,d}")

            # Raw signal stats
            mean, stdev, time_to_elm_quantiles = \
                self.get_statistics_for_global_stage(signal_windows=global_sw_metadata_list)
            assert mean and stdev and time_to_elm_quantiles

            if sub_stage == 'train' and not self.elm_raw_signal_mean:
                self.zprint(f"    Using {sub_stage.upper()} for standardizing mean and stdev")
                self.elm_raw_signal_mean = mean.item()
                self.elm_raw_signal_stdev = stdev.item()
                self.time_to_elm_quantiles = time_to_elm_quantiles
                self.save_hyperparameters({
                    'elm_raw_signal_mean': self.elm_raw_signal_mean,
                    'elm_raw_signal_stdev': self.elm_raw_signal_stdev,
                    'time_to_elm_quantiles': self.time_to_elm_quantiles,
                })
            else:
                assert self.elm_raw_signal_mean and self.elm_raw_signal_stdev and self.time_to_elm_quantiles

        # restrict data according to quantiles
        if self.is_global_zero:
            if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
                time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in global_sw_metadata_list])
                time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
                if self.contrastive_learning:
                    self.zprint(f"  Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                    for i in np.arange(len(global_sw_metadata_list)-1, -1, -1, dtype=int):
                        if (global_sw_metadata_list[i]['time_to_elm'] > time_to_elm_min) and \
                            (global_sw_metadata_list[i]['time_to_elm'] < time_to_elm_max):
                            global_sw_metadata_list.pop(i)
                else:
                    self.zprint(f"  Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                    for i in np.arange(len(global_sw_metadata_list)-1, -1, -1, dtype=int):
                        if (global_sw_metadata_list[i]['time_to_elm'] < time_to_elm_min) or \
                            (global_sw_metadata_list[i]['time_to_elm'] > time_to_elm_max):
                            global_sw_metadata_list.pop(i)
                n_signal_windows = len(global_sw_metadata_list)
                print(f"Restricted global signal windows: {n_signal_windows:,d}")
                print(f"Restricted global steps per epoch {n_signal_windows/self.batch_size:,.1f}")

        return global_sw_metadata_list

    def prepare_rank_data_for_stage(self, sub_stage: str, global_sw_metadata_list: list) -> None:
        self.zprint("\u2B1C " + f"Preparing rank data for stage: {sub_stage.upper()}")

        # split signal windows by rank
        rankwise_sw_split = np.array_split(global_sw_metadata_list, self.trainer.world_size)
        sw_for_rank = list(rankwise_sw_split[self.trainer.global_rank])
        elms_for_rank = np.unique(np.array(
            [item['elm_index'] for item in sw_for_rank],
            dtype=int,
        ))
        shots_for_rank = np.unique(np.array(
            [item['shot'] for item in sw_for_rank],
            dtype=int,
        ))
        self.barrier()
        self.rprint(f"Shots/ELMs/SigWin/BatchesPerEpoch: {len(shots_for_rank):,d}/{len(elms_for_rank):,d}/{len(sw_for_rank):,d}/{len(sw_for_rank)/(self.batch_size/self.world_size):,.1f}")
        self.barrier()

        # get rank-wise ELM signals
        signals_for_rank = {}
        with h5py.File(self.elm_data_file) as root:
            for elm_index in elms_for_rank:
                elm_group: h5py.Group = root['elms'][f"{elm_index:06d}"]
                signals = np.array(elm_group["bes_signals"], dtype=np.float32)  # (64, <time>)
                signals = np.transpose(signals).reshape(1, -1, 8, 8)  # reshape to (time, pol, rad)
                # normalized signals
                signals_for_rank[elm_index] = (signals - self.elm_raw_signal_mean) / self.elm_raw_signal_stdev
        assert len(signals_for_rank) == len(elms_for_rank)

        # rank-wise datasets
        if sub_stage in ['train', 'validation', 'test']:
            self.elm_datasets[sub_stage] = ELM_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                time_to_elm_quantiles=self.time_to_elm_quantiles,
                sw_list=sw_for_rank,
                signal_list=signals_for_rank,
                quantile_min=self.time_to_elm_quantile_min,
                quantile_max=self.time_to_elm_quantile_max,
                contrastive_learning=self.contrastive_learning,
            )
            assert len(self.elm_datasets[sub_stage]) == len(sw_for_rank)
        
        if sub_stage in ['test', 'predict']:
            pass

    def split_global_data(self):
        self.zprint("\u2B1C Splitting global data into train/validation/test sets")
        if 'train' in self.global_elm_split and len(self.global_elm_split['train'])>0:
            assert 'train' in self.global_shot_split and len(self.global_shot_split['train']) > 0
            self.zprint("  Global data already split, returning")
            return
        assert len(self.global_elm_split) == 0 and len(self.global_shot_split) == 0
        self.zprint(f"  Initializing RNG with seed: {self.seed}")
        rng = np.random.default_rng(self.seed)
        with h5py.File(self.elm_data_file, 'r') as root:
            datafile_shots = set([int(shot_key) for shot_key in root['shots']])
            datafile_shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(datafile_shots ^ datafile_shots_from_elms) == 0
            datafile_shots = list(datafile_shots)
            datafile_elms = [int(elm_key) for elm_key in root['elms']]
            self.zprint(f"  ELMs/shots in HDF5 file: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            # limit max ELMs
            if self.max_elms and len(datafile_elms) > self.max_elms:
                self.zprint(f"  Shuffling ELMs and limiting to {self.max_elms:,d} global ELMs")
                rng.shuffle(datafile_elms)
                datafile_elms = datafile_elms[:self.max_elms]
                datafile_shots = set([int(root['elms'][f"{elm_index:06d}"].attrs['shot']) for elm_index in datafile_elms])
                datafile_shots = list(datafile_shots)
                self.zprint(f"  Global ELMs/shots for training: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            # shuffle shots in dataset
            self.zprint(f"  Shuffling global shots")
            rng.shuffle(datafile_shots)
            if len(datafile_shots) > 5:
                self.zprint(f"  Global shot order: " + ', '.join(map(str, datafile_shots[:5])) + " ...")
            # order ELMs by shuffled shots
            self.zprint(f"  Ordering global ELMs by global shot order")
            new_datafile_elms = []
            datafile_elms = sorted(datafile_elms)
            for shot in datafile_shots:
                start_new_shot = True
                for i_elm in datafile_elms:
                    if root['elms'][f"{i_elm:06d}"].attrs['shot'] == shot:
                        new_datafile_elms.append(i_elm)
                        start_new_shot = False
                    else:
                        if start_new_shot == False:
                            break
            datafile_elms = new_datafile_elms

            self.zprint("  Splitting global shots into train/validation/test data sets")
            n_test_shots = int(self.fraction_test * len(datafile_shots))
            n_validation_shots = int(self.fraction_validation * len(datafile_shots))
            self.global_shot_split['test'], self.global_shot_split['validation'], self.global_shot_split['train'] = \
                np.split(datafile_shots, [n_test_shots, n_test_shots+n_validation_shots])
            
            self.zprint(f"  Splitting global ELMs by global shot split")
            for stage in ['train', 'validation', 'test']:
                self.global_elm_split[stage] = [
                    i_elm for i_elm in datafile_elms
                    if root['elms'][f"{i_elm:06d}"].attrs['shot'] in self.global_shot_split[stage]
                ]
                self.zprint(f"  Stage {stage.upper()}: Global shot count {self.global_shot_split[stage].size}")
                if len(self.global_shot_split[stage]) > 5:
                    self.zprint(f"  Stage {stage.upper()}: Global shot order: " + ', '.join(map(str, self.global_shot_split[stage][:5])) + " ...")
                self.zprint(f"  Stage {stage.upper()}: Global ELM count {len(self.global_elm_split[stage]):,d}")
                if len(self.global_elm_split[stage]) > 5:
                    self.zprint(f"  Stage {stage.upper()}: Global ELM order: " + ', '.join(map(str, self.global_elm_split[stage][:5])) + " ...")

            assert len(self.global_elm_split['train']) > 0
            assert len(self.global_elm_split['validation']) > 0

    def get_statistics_for_global_stage(
            self, 
            signal_windows: list[dict],
    ) -> tuple:
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_interval = np.max([self.stride_factor, len(signal_windows)//int(50e3)])
        last_elm_index = -1
        with h5py.File(self.elm_data_file) as root:
            for elm_dict in signal_windows[::stat_interval]:
                elm_index = elm_dict['elm_index']
                if elm_index != last_elm_index:
                    elm_event: h5py.Group = root['elms'][f'{elm_index:06d}']
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)  # (64, <time>)
                    signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (time, pol, rad)
                last_elm_index = elm_index
                i_t0 = elm_dict['i_t0']
                signal_window = signals[i_t0: i_t0 + self.signal_window_size, :, :]
                assert signal_window.shape[0] == self.signal_window_size
                signal_min = np.min([signal_min, signal_window.min()])
                signal_max = np.max([signal_max, signal_window.max()])
                hist, bin_edges = np.histogram(
                    signal_window,
                    bins=n_bins,
                    range=(-10.4, 10.4),
                )
                cummulative_hist += hist
        bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
        stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
        exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
        # time-to-ELM quantiles
        time_to_elm_list = [e['time_to_elm'] for e in signal_windows]
        quantiles = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        quantile_values = np.quantile(time_to_elm_list, quantiles)
        time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}

        self.zprint(f"  Raw signals min {signal_min:.2f} max {signal_max:.2f} mean {mean:.2f} stdev {stdev:.2f} exkurt {exkurt:.2f}")
        self.zprint(f"  Time-to-ELM quantiles:")
        for q, qval in time_to_elm_quantiles.items():
            self.zprint(f"    Quantile {q:.2f}: {qval:.1f} ms")

        return (mean.item(), stdev.item(), time_to_elm_quantiles)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.elm_dataloaders('train')

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.elm_dataloaders('validation')

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.elm_dataloaders('test')

    def predict_dataloader(self) -> None:
        pass

    def elm_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        # sampler = (
        #     torch.utils.data.RandomSampler(data_source=self.elm_datasets[stage])
        #     if stage == 'train'
        #     else torch.utils.data.SequentialSampler(data_source=self.elm_datasets[stage])
        # )
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.elm_datasets[stage],
            # num_replicas=1,
            # rank=0,
            shuffle=True if stage=='train' else False,
            seed=int(np.random.default_rng().integers(0, 2**32-1)),
            drop_last=True if stage=='train' else False,
        )
        batch_size_reduction_factor = (
            min(3, self.trainer.current_epoch//self.epochs_per_batch_size_reduction) 
            if stage=='train' 
            else 0
        )
        default_batch_size_per_rank = self.batch_size // self.trainer.world_size
        new_batch_size_per_rank = default_batch_size_per_rank // (2**batch_size_reduction_factor)
        if self._modified_batch_size_per_rank and self._modified_batch_size_per_rank != new_batch_size_per_rank:
            self.zprint(f"New global batchsize: {new_batch_size_per_rank*self.trainer.world_size}")
        self._modified_batch_size_per_rank = new_batch_size_per_rank
        return torch.utils.data.DataLoader(
            dataset=self.elm_datasets[stage],
            sampler=sampler,
            batch_size=self._modified_batch_size_per_rank,  # batch size per rank
            num_workers=self.num_workers,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=True if stage=='train' else False,
            # persistent_workers=False,
        )

    def get_state_dict(self) -> dict:
        state_dict = {item: getattr(self, item) for item in self.state_items}
        return state_dict

    def load_state_dict(self, state: dict) -> None:
        for item in self.state_items:
            setattr(self, item, state[item])

    def save_state_dict(self):
        if self.is_global_zero:
            self.zprint("Saving `state_dict`")
            state_dict = self.get_state_dict()
            torch.save(state_dict, Path(self.log_dir)/'state_dict.pt')

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        if self.trainer.world_size > 1:
            self.barrier()
            print(f"  Rank {self.trainer.global_rank}: {text}")
            self.barrier()
        else:
            print(f"  {text}")

    def barrier(self):
        if self.world_size > 1:
            self.trainer.strategy.barrier()

    def broadcast(self, obj):
        if self.trainer.world_size > 0:
            obj = self.trainer.strategy.broadcast(obj)
        return obj


@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(_Base_Class, torch.utils.data.Dataset):
    signal_window_size: int = 0
    sw_list: list = None # global signal window data mapping to dataset index
    signal_list: dict = None # rank-wise signals (map to ELM indices)
    time_to_elm_quantiles: dict[float, float] = None
    quantile_min: float = None
    quantile_max: float = None
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
        return signal_window, time_to_elm, quantile_binary_label

def main(
        elm_data_file: str|Path = None,
        elm_classifier: bool = True,
        max_elms: int = None,
        signal_window_size: int = 512,
        experiment_name: str = 'experiment_default',
        restart_trial_name: str = None,
        wandb_id: str = None,
        # model
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler_patience: int = 100,
        lr_warmup_epochs: int = 5,
        monitor_metric = None,
        use_optimizer: str = 'SGD',
        no_bias: bool = False,
        batch_norm: bool = True,
        feature_model_layers = None,
        mlp_task_models = None,
        # loggers
        log_freq: int = 100,
        use_wandb: bool = False,
        # callbacks
        early_stopping_min_delta: float = 1e-3,
        early_stopping_patience: int = 100,
        # trainer
        max_epochs = 2,
        gradient_clip_val: float = None,
        gradient_clip_algorithm = None,
        skip_train: bool = False,
        skip_data: bool = False,
        precision = None,
        # data
        batch_size: int = 64,
        fraction_validation: float = 0.1,
        fraction_test: float = 0.0,
        num_workers: int = 4,
        time_to_elm_quantile_min: float = None,
        time_to_elm_quantile_max: float = None,
        contrastive_learning: bool = True,
        min_pre_elm_time: float = None,
        fir_bp_low: float = None,
        fir_bp_high: float = None,
        epochs_per_batch_size_reduction: int = 100,
) -> tuple:

    ### SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    world_rank = int(os.getenv("SLURM_PROCID", default=0))
    local_rank = int(os.getenv("SLURM_LOCALID", default=0))
    node_rank = int(os.getenv("SLURM_NODEID", default=0))

    print(f"Rank {world_rank} of world size {world_size} (local rank {local_rank} on node {node_rank})")

    is_global_zero = world_rank == 0

    def zprint(text):
        if is_global_zero:
            print(text)

    def rprint(text):
        if world_size==1:
            print(text)
        else:
            print(f"Global rank {world_rank}: {text}")

    ### model
    zprint("\u2B1C Creating model")
    lit_model = Model(
        elm_classifier=elm_classifier,
        signal_window_size=signal_window_size,
        lr=lr,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_warmup_epochs=lr_warmup_epochs,
        weight_decay=weight_decay,
        # dropout_percent=dropout_percent,
        monitor_metric=monitor_metric,
        use_optimizer=use_optimizer,
        is_global_zero=is_global_zero,
        world_size=world_size,
        world_rank=world_rank,
        # lr_layerwise_decrement=layerwise_lr_decrement,
        no_bias=no_bias,
        batch_norm=batch_norm,
        feature_model_layers=feature_model_layers,
        mlp_task_models=mlp_task_models,
    )

    monitor_metric = lit_model.monitor_metric

    zprint("\u2B1C Model Summary:")
    zprint(ModelSummary(lit_model, max_depth=-1))

    ### callbacks
    zprint("\u2B1C Creating callbacks")
    metric_mode = 'min' if 'loss' in monitor_metric else 'max'
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            monitor=monitor_metric,
            mode=metric_mode,
            save_last=True,
        ),
        # DeviceStatsMonitor(),
        EarlyStopping(
            monitor=monitor_metric,
            mode=metric_mode,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            log_rank_zero_only=True,
            verbose=True,
        ),
    ]

    ### loggers
    zprint("\u2B1C Creating loggers")
    loggers = []
    experiment_dir = Path(experiment_name).absolute()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if restart_trial_name:
        trial_name = restart_trial_name
    else:
        datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
        trial_name = f"r{slurm_identifier}_{datetime_str}" if slurm_identifier else f"r{datetime_str}"
    tb_logger = TensorBoardLogger(
        save_dir=experiment_dir.parent, # parent directory of the experiment directory
        name=experiment_name,  # experiment directory name
        version=trial_name,  # trial directory name within the experiment directory
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    zprint(f"Tensorboard trial name: {trial_name}")
    if use_wandb:
        wandb.login()
        wandb_save_dir = experiment_dir
        wandb_project = experiment_name
        wandb_name = trial_name
        wandb_logger = WandbLogger(
            save_dir=wandb_save_dir,
            project=wandb_project,
            name=wandb_name,
            id=wandb_id if wandb_id else None,
            resume='must' if restart_trial_name else None,
        )
        wandb_logger.watch(
            model=lit_model, 
            log='all', 
            log_freq=log_freq,
        )
        loggers.append(wandb_logger)
        zprint(f"WandB ID/version: {wandb_logger.version}")

    zprint(f"World size {world_size} on {num_nodes} node(s)")
    rprint(f"Local rank {local_rank} on node {node_rank}")

    ### initialize trainer
    zprint("\u2B1C Creating Trainer")
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
        enable_progress_bar = False,
        enable_model_summary = False,
        precision = precision,
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            static_graph=True,
        ) if world_size>1 else 'auto',
        num_nodes = num_nodes,
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
    )

    assert trainer.node_rank == node_rank
    assert trainer.world_size == world_size
    assert trainer.local_rank == local_rank
    assert trainer.global_rank == world_rank
    assert trainer.is_global_zero == is_global_zero
    assert trainer.log_dir == tb_logger.log_dir

    ckpt_path=experiment_dir/restart_trial_name/'checkpoints/last.ckpt' if restart_trial_name else None

    ### data
    if not skip_data:
        zprint("\u2B1C Creating data module")
        if ckpt_path:
            assert ckpt_path.exists(), f"Checkpoint does not exist: {ckpt_path}"
            print(f"Loading data from checkpoint: {ckpt_path}")
            lit_datamodule = Data.load_from_checkpoint(checkpoint_path=ckpt_path)
            state_dict_file = experiment_dir/restart_trial_name/'state_dict.pt'
            assert state_dict_file.exists(), f"State dict file does not exist: {state_dict_file}"
            print(f"Loading state_dict from: {state_dict_file}")
            state_dict = torch.load(state_dict_file, weights_only=False)
            lit_datamodule.load_state_dict(state_dict)
        else:
            lit_datamodule = Data(
                signal_window_size=signal_window_size,
                log_dir=trainer.log_dir,
                elm_data_file=elm_data_file,
                max_elms=max_elms,
                batch_size=batch_size,
                fraction_test=fraction_test,
                fraction_validation=fraction_validation,
                num_workers=num_workers,
                time_to_elm_quantile_min=time_to_elm_quantile_min,
                time_to_elm_quantile_max=time_to_elm_quantile_max,
                contrastive_learning=contrastive_learning,
                is_global_zero=is_global_zero,
                world_size=world_size,
                world_rank=world_rank,
                min_pre_elm_time=min_pre_elm_time,
                fir_bp_low=fir_bp_low,
                fir_bp_high=fir_bp_high,
                epochs_per_batch_size_reduction=epochs_per_batch_size_reduction,
            )

    if not skip_train and not skip_data:
        zprint("\u2B1C Begin Trainer.fit()")
        trainer.fit(
            model=lit_model, 
            datamodule=lit_datamodule,
            ckpt_path=ckpt_path,
        )
        if fraction_test:
            trainer.test(
                model=lit_model, 
                datamodule=lit_datamodule,
                ckpt_path=ckpt_path,
            )

    if use_wandb:
        wandb_id = wandb_logger.version
        wandb.finish()
        if is_global_zero:
            print(f"WB ID: {wandb_id}")
    else:
        wandb_id = None

    if is_global_zero:
        print(f"TB trial name: {trial_name}")

    return (trial_name, wandb_id)

if __name__=='__main__':
    main(
        restart_trial_name='',
        wandb_id='',
        elm_data_file='/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/small_data_200.hdf5',
        signal_window_size=512,
        max_elms=20,
        max_epochs=2,
        batch_size=128,
        lr=1e-3,
        num_workers=1,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        no_bias=True,
        batch_norm=False,
        # time_to_elm_quantile_min=0.4,
        # time_to_elm_quantile_max=0.6,
        # contrastive_learning=True,
        # fir_bp_low=5,
        # fir_bp_high=250,
        # skip_data=True,
        # skip_train=True,
        # use_wandb=True,
    )