from pathlib import Path
import dataclasses
from datetime import datetime
from typing import OrderedDict, Sequence
import os
import time
import re

import lightning
import lightning.pytorch
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
    LearningRateMonitor, EarlyStopping, ModelCheckpoint, BackboneFinetuning
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.combined_loader import CombinedLoader

import ml_data

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
    signal_window_size: int = 256

    def __post_init__(self):
        assert np.log2(self.signal_window_size).is_integer(), \
            'Signal window must be power of 2'

        self.world_size = int(os.getenv("SLURM_NTASKS", default=1))
        self.world_rank = int(os.getenv("SLURM_PROCID", default=0))
        self.is_global_zero = self.world_rank == 0

        if self.world_rank > 0:
            assert self.world_size > 1

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        if self.world_size > 1:
            print(f"{text} (Rank {self.world_rank})")
        else:
            print(f"{text}")


@dataclasses.dataclass(eq=False)
class Model(_Base_Class, LightningModule):
    # elm_classifier: bool = True
    # conf_classifier: bool = False
    lr: float = 1e-3
    deepest_layer_lr_factor: float = 0.1
    lr_scheduler_patience: int = 100
    lr_scheduler_threshold: float = 1e-3
    lr_warmup_epochs: int = 10
    weight_decay: float = 1e-6
    leaky_relu_slope: float = 2e-2
    monitor_metric: str = None
    use_optimizer: str = 'adam'
    elm_loss_weight: float = None
    conf_loss_weight: float = None
    no_bias: bool = False
    batch_norm: bool = True
    dropout: float = 0.0
    feature_model_layers: Sequence[dict[str, LightningModule]] = None
    mlp_tasks: dict[str, Sequence] = None
    backbone_model_path: str|Path = None
    backbone_first_n_layers: int = None

    def __post_init__(self):

        # init superclasses
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()
        self.trainer: lightning.pytorch.Trainer = None

        if self.is_global_zero:
            print_fields(self)

        # only batch_norm or dropout, not both
        assert not (self.batch_norm and self.dropout)

        # input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.feature_model: LightningModule = None
        self.feature_space_size: int = None
        self.make_feature_model()
        assert self.feature_space_size

        # default task sub-model and metrics
        if self.mlp_tasks is None:
            self.mlp_tasks = {
                'elm_class': [self.feature_space_size, 16, 1],
            }
        
        # set input layer size
        for task in self.mlp_tasks:
            if not self.mlp_tasks[task][0]:
                self.mlp_tasks[task][0] = self.feature_space_size

        # create MLP configs
        self.task_configs = {}
        for task, mlp_layers in self.mlp_tasks.items():
            self.task_configs[task] = {}
            self.task_configs[task]['layers'] = mlp_layers
            if 'class' in task:
                self.task_configs[task]['metrics'] = {
                        'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
                        'f1_score': sklearn.metrics.f1_score,
                        'precision_score': sklearn.metrics.precision_score,
                        'recall_score': sklearn.metrics.recall_score,
                        'mean_stat': torch.mean,
                        'std_stat': torch.std,
                    }
                self.task_configs[task]['monitor_metric'] = 'f1_score/val'

        # make task sub-models
        self.task_names = list(self.task_configs.keys())
        self.task_metrics: dict[str, dict] = {}
        self.task_models: torch.nn.ModuleDict = torch.nn.ModuleDict()
        for task_name, task_dict in self.task_configs.items():
            self.zprint(f'Task sub-model: {task_name}')
            task_layers: tuple[int] = task_dict['layers']
            task_metrics: dict = task_dict['metrics']
            self.task_models[task_name] = self.make_mlp_classifier(mlp_layers=task_layers)
            self.task_metrics[task_name] = task_metrics.copy()
            if self.monitor_metric is None:
                # if not specified, use `monitor_metric` from first task
                self.monitor_metric = f"{task_name}/{task_dict['monitor_metric']}"
        self.save_hyperparameters({'monitor_metric': self.monitor_metric})

        self.zprint(f"Total model parameters: {self.param_count(self):,d}")

        # # sub-model: Confinement mode multi-class classifier
        # if self.conf_classifier:
        #     task_name = 'conf_classifier'
        #     self.zprint(f"Task {task_name}")
        #     self.task_models[task_name] = self.make_mlp_classifier(n_out=4)
        #     self.task_metrics[task_name] = {
        #         'ce_loss': torch.nn.functional.cross_entropy,
        #         'f1_score': sklearn.metrics.f1_score,
        #         'precision_score': sklearn.metrics.precision_score,
        #         'recall_score': sklearn.metrics.recall_score,
        #         'mean_stat': lambda t: torch.abs(torch.mean(t)), #torch.mean,
        #         'std_stat': torch.std,
        #     }
        #     if self.monitor_metric is None:
        #         self.monitor_metric = f'{task_name}/f1_score/val'

        self.initialize_parameters()
        if self.backbone_model_path and self.backbone_first_n_layers:
            self.backbone_transfer_learning()

    def make_feature_model(self) -> None:
        self.zprint("Feature space sub-model")
        if self.feature_model_layers is None:
            self.feature_model_layers = (
                {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias':True},
                {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1, 'bias':True},
                {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias':True},
            )
        n_feature_layers = len(self.feature_model_layers)
        feature_layer_dict = OrderedDict()
        data_shape = self.input_data_shape
        self.zprint(f"  Input data shape: {data_shape}  (size {np.prod(data_shape)})")
        previous_out_channels: int = None
        for i_layer, layer in enumerate(self.feature_model_layers):
            in_channels: int = 1 if i_layer==0 else previous_out_channels
            # batch norm before conv layer (except input layer)
            if i_layer > 0 and self.batch_norm:
                layer_name = f"L{i_layer:02d}_BatchNorm"
                feature_layer_dict[layer_name] = torch.nn.BatchNorm3d(in_channels)
                self.zprint(f"  {layer_name} (regularization)")
            # conv layer
            conv_layer_name = f"L{i_layer:02d}_Conv"
            bias = layer['bias'] and (self.no_bias == False)
            conv = torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
                bias=bias,
            )
            feature_layer_dict[conv_layer_name] = conv
            n_params = self.param_count(conv)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            self.zprint(f"  {conv_layer_name} kern {conv.kernel_size}  stride {conv.stride}  bias {bias}  out_ch {conv.out_channels}  param {n_params:,d}  output {data_shape} (size {np.prod(data_shape)})")
            previous_out_channels = layer['out_channels']
            # LeakyReLU layer
            relu_layer_name = f"L{i_layer:02d}_LeakyReLU"
            feature_layer_dict[relu_layer_name] = torch.nn.LeakyReLU(self.leaky_relu_slope)
            self.zprint(f"  {relu_layer_name} (activation)")
            # dropout after activation (except after last layer)
            if i_layer < n_feature_layers - 1 and self.dropout:
                feature_layer_dict[f"L{i_layer:02d}_Dropout"] = torch.nn.Dropout3d(self.dropout)
                self.zprint(f"  {layer_name} (regularization)")

        layer_name = f'L{n_feature_layers - 1:02d}_Flatten'
        feature_layer_dict[layer_name] = torch.nn.Flatten()
        self.zprint(f"  {layer_name}  (flatten to vector)")
        self.feature_model = torch.nn.Sequential(feature_layer_dict)
        self.feature_space_size = self.feature_model(torch.zeros(self.input_data_shape)).numel()

        self.zprint(f"    Feature sub-model parameters: {self.param_count(self.feature_model):,d}")
        self.zprint(f"    Feature space size: {self.feature_space_size}")

    def make_mlp_classifier(
            self,
            mlp_layers: Sequence[int] = None,
    ) -> torch.nn.Module:
        self.zprint("MLP classifier sub-model")
        mlp_layer_dict = OrderedDict()
        assert mlp_layers
        n_mlp_layers = len(mlp_layers)
        n_feature_layers = len(self.feature_model_layers)

        for i_mlp_layer in range(n_mlp_layers-1):
            # batch norm
            i_layer = n_feature_layers + i_mlp_layer
            if self.batch_norm:
                layer_name = f"L{i_layer:02d}_BatchNorm"
                mlp_layer_dict[layer_name] = torch.nn.BatchNorm1d(mlp_layers[i_mlp_layer])
                self.zprint(f'  {layer_name} (regularization)')
            # fully-connected layer
            layer_name = f"L{i_layer:02d}_FC"
            bias = (True and (self.no_bias==False)) if i_mlp_layer<n_mlp_layers-2 else False
            mlp_layer = torch.nn.Linear(
                in_features=mlp_layers[i_mlp_layer],
                out_features=mlp_layers[i_mlp_layer+1],
                bias=bias,
            )
            mlp_layer_dict[layer_name] = mlp_layer
            n_params = self.param_count(mlp_layer)
            self.zprint(f"  {layer_name}  bias {bias}  in_features {mlp_layer.in_features}  out_features {mlp_layer.out_features}  parameters {n_params:,d}")
            # leaky relu
            if i_mlp_layer+1 != n_mlp_layers-1:
                layer_name = f"L{i_layer:02d}_LeakyReLU"
                mlp_layer_dict[layer_name] = torch.nn.LeakyReLU(self.leaky_relu_slope)
                self.zprint(f"  {layer_name} (activation)")

        mlp_classifier = torch.nn.Sequential(mlp_layer_dict)

        self.zprint(f"    MLP sub-model parameters: {self.param_count(mlp_classifier):,d}")

        return mlp_classifier

    def initialize_parameters(self) -> None:
        if not self.is_global_zero:
            return
        # initialize all biases to 0
        for param_name, param in self.named_parameters():
            if param_name.endswith("bias"):
                param.data.fill_(0)
        good_init = False
        while good_init == False:
            self.zprint("Initializing model to uniform random weights (biases=0)")
            good_init = True
            for param_name, param in self.named_parameters():
                if not param_name.endswith('weight'):
                    continue
                if 'BatchNorm' in param_name:
                    param.data.fill_(1)
                else:
                    n_in = np.prod(param.shape[1:])
                    sqrt_k = np.sqrt(3 / n_in)
                    param.data.uniform_(-sqrt_k, sqrt_k)
                    self.zprint(f"  {param_name}: initialized to uniform +- {sqrt_k:.1e} n*var: {n_in*torch.var(param.data):.3f} (n {param.data.numel()})")
            random_batch_input = {
                task: [torch.randn(
                    size=[512]+list(self.input_data_shape[1:]),
                    dtype=torch.float32,
                )]
                for task in self.task_names
            }
            example_batch_output = self(random_batch_input)
            for task_name, task_output in example_batch_output.items():
                if task_output.mean().abs() / task_output.std() > 0.25:
                    good_init = False
                    self.zprint("Large mean for random inputs; re-initializing...")
                    break
        # good init is satisfied
        for task_name, task_output in example_batch_output.items():
            self.zprint(f"Batch evaluation (batch_size=512) with randn() data")
            self.zprint(f"  Task {task_name} output shape: {task_output.shape}")
            self.zprint(f"  Task {task_name} output mean {task_output.mean():.4f} stdev {task_output.std():.4f} min/max {task_output.min():.3f}/{task_output.max():.3f}")

    def backbone_transfer_learning(self) -> None:
        if not self.is_global_zero or not self.backbone_model_path or not self.backbone_first_n_layers:
            return
        self.backbone_model_path = Path(self.backbone_model_path) / 'checkpoints'/'last.ckpt'
        assert self.backbone_model_path.exists(), f"Transfer model not found: {self.backbone_model_path}"
        self_param_names = [param_name for param_name, _ in self.named_parameters()]
        source_model = torch.load(
            self.backbone_model_path,
            weights_only=False,
        )
        src_state_dict = source_model['state_dict']
        for param_name in list(src_state_dict.keys()):
            if not param_name.endswith(('bias', 'weight')):
                src_state_dict.pop(param_name)
                continue
            i_layer = int(re.findall(r'L(\d+)_', param_name)[0])
            if i_layer >= self.backbone_first_n_layers:
                src_state_dict.pop(param_name)
        self.zprint(f'params to copy from src to self:')
        for param_name in src_state_dict:
            self.zprint(f'  {param_name}')
            assert param_name in self_param_names, \
                f"{param_name} in source model, but not in self model"
        _ = self.load_state_dict(
            state_dict=src_state_dict, 
            strict=False,
        )
        def flatten_modules(module: torch.nn.Module):
            for name, submodule in module.named_children():
                if len(list(submodule.children())) == 0:
                    yield name, submodule
                else:
                    self.zprint(f"Traversing into {name}")
                    yield from flatten_modules(submodule)
        module_dict = dict(flatten_modules(self))
        self.backbone = torch.nn.Sequential()
        for module_name, module in module_dict.items():
            if self.param_count(module)==0: continue
            res = re.findall(r'L(\d+)_', module_name)
            if not res: continue
            assert len(res) < 2
            i_layer = int(res[0])
            if i_layer < self.backbone_first_n_layers:
                self.zprint(f"Adding module {module_name} to backbone")
                self.backbone.add_module(module_name, module)
        # print('missing from src:', missing)
        # print('unexpected in src:', unexpected)
        # # for self_mod_name, self_mod in self.named_modules():
        # #     print(self_mod.named_parameters())
        # #     for src_param_name, src_param in source_model['state_dict'].items():
        # #         if src_param_name.startswith(self_mod_name):
        # #             print(self_mod_name, self_mod.shape, src_param.shape)

    def configure_optimizers(self):
        self.zprint("\u2B1C " + f"Creating {self.use_optimizer.upper()} optimizer")
        self.zprint(f"  lr: {self.lr:.1e}")
        self.zprint(f"  lr for deepest layer: {self.deepest_layer_lr_factor * self.lr:.1e}")
        self.zprint(f"  Warmup epochs: {self.lr_warmup_epochs}")
        self.zprint(f"  Reduce on plateau threshold {self.lr_scheduler_threshold:.1e} and patience {self.lr_scheduler_patience:d}")

        optim_kwargs = {
            'params': self.make_parameter_list(),
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
        optimizer = None
        if self.use_optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(momentum=0.2, **optim_kwargs)
        elif self.use_optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(**optim_kwargs)
        else:
            raise ValueError(f"Unknown optimizer {self.use_optimizer}")

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            min_lr=1e-6,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.05,
            total_iters=self.lr_warmup_epochs,
        )
        return (
            [optimizer],
            [
                {'scheduler': plateau_scheduler, 'monitor': self.monitor_metric},
                warmup_scheduler, 
            ],
        )

    def make_parameter_list(self) -> list[dict]:
        parameter_list = []
        n_feature_layers = 0
        for layer_name, _ in self.named_modules():
            if not layer_name.endswith(('Conv','FC')):
                continue
            n_feature_layers += 1
        lrs_for_feature_layers = np.logspace(
            np.log10(self.lr * self.deepest_layer_lr_factor),
            np.log10(self.lr),
            n_feature_layers,
        )
        for param_name, params in self.named_parameters():
            param_dict = {}
            # no weight_decay for biases
            if 'bias' in param_name:
                param_dict['weight_decay'] = 0.
            if 'BatchNorm' in param_name:
                # default lr for BatchNorm
                param_dict['lr'] = self.lr
            else:
                # lr for Conv and FC layers
                assert 'Conv' in param_name or 'FC' in param_name
                assert param_name.endswith(('weight','bias'))
                i_layer = int(re.findall(r'L(\d+)_', param_name)[0])
                if param_name.endswith('weight'):
                    param_dict['lr'] = lrs_for_feature_layers[i_layer]
                else:              
                    param_dict['lr'] = lrs_for_feature_layers[i_layer]/10
            optimizer_params = ''.join([f'  {key} {value:.1e}' for key, value in param_dict.items()])
            self.zprint(f"  {param_name}: {optimizer_params}")
            param_dict['params'] = params
            parameter_list.append(param_dict)

        return parameter_list

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
            task_outputs: torch.Tensor = model_outputs[task]
            metrics = self.task_metrics[task]
            if task == 'elm_class' and dataloader_idx in [None, 0]:
                labels: torch.Tensor = batch[task][1][0.5] if isinstance(batch, dict) else batch[1][0.5]
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
            batch: dict|list, 
    ) -> dict[str, torch.Tensor]:
        results = {}
        for task_name, task_model in self.task_models.items():
            x = (
                batch[task_name][0]
                if isinstance(batch, dict) 
                else batch[0]
            )
            features = self.feature_model(x)
            results[task_name] = task_model(features)
        return results

    def on_fit_start(self):
        self.t_fit_start = time.time()
        self.zprint(f"**** Fit start with max epochs {self.trainer.max_epochs} ****")

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step
        self.current_max_lr = np.max(self.trainer.lr_scheduler_configs[-1].scheduler.get_last_lr())
        self.current_batch_size = self.trainer.train_dataloader['elm_class'].batch_size
        # lr = self.optimizers.get_last_lr()
        # lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        # print(self.trainer.lr_scheduler_configs[-1].scheduler.get_last_lr())
        # if isinstance(self.warmup_lr_factors, dict):
        #     if self.current_epoch not in self.warmup_lr_factors:
        #         return
            
        #     lr: float = None
        #     for key, value in reversed(self.lr.items()):
        #         if self.current_epoch >= key:
        #             lr = value
        #             break
        #     assert isinstance(lr, float)
        #     optimizer: torch.optim.SGD = self.optimizers()
        #     param_list = self.make_parameter_list(largest_lr=lr)
        #     # self.zprint(param_list)
        #     for param_group in optimizer.param_groups:
        #         for param_dict in param_list:
        #             if torch.equal(param_dict['params'], param_group['params']):
        #                 print(param_group['lr'], param_dict['lr'])
        #                 break

    # def on_train_batch_start(self, batch, batch_idx):
    #     # if batch_idx % 25 == 0:
    #     #     self.rprint(f"Train batch start: batch {batch_idx}")
    #     pass

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     # self.zprint(f"Validation batch start: batch {batch_idx}, dataloader {dataloader_idx}")
    #     pass

    def on_train_epoch_end(self):
        if self.is_global_zero and self.global_step > 0:
            epoch_time = time.time() - self.t_train_epoch_start
            global_time = time.time() - self.t_fit_start
            epoch_steps = self.global_step-self.s_train_epoch_start
            logged_metrics = self.trainer.logged_metrics
            line =  f"Ep {self.current_epoch+1:03d}"
            line += f"  lr {self.current_max_lr:.1e}"
            line += f" bs {self.current_batch_size}"
            line += f" tr/val loss {logged_metrics['sum_loss/train']:.3f}/"
            line += f"{logged_metrics['sum_loss/val']:.3f}"
            line += f" ep/gl steps {epoch_steps:,d}/{self.global_step:,d}"
            line += f" ep/gl time (min): {epoch_time/60:.1f}/{global_time/60:.1f}" 
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

    @staticmethod
    def param_count(model: LightningModule) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    elm_data_file: str|Path = None
    # elm_classifier: bool = True
    confinement_data_file: str|Path = None
    # conf_classifier: bool = False
    tasks: Sequence[str] = ('elm_class',)
    log_dir: str|Path = None
    max_elms: int = None
    batch_size: int|dict = 128
    stride_factor: int = 8
    num_workers: int = 2
    outlier_value: float = 6
    fraction_validation: float = 0.1
    fraction_test: float = 0.0
    use_random_data: bool = False
    seed: int = None  # seed for ELM index shuffling; must be same across processes
    time_to_elm_quantile_min: float = None
    time_to_elm_quantile_max: float = None
    contrastive_learning: bool = False
    min_pre_elm_time: float = None
    fir_bp: Sequence[float] = None  # bandpass filter cut-on and cut-off frequencies in kHz
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
    max_confinement_event_length: int = None

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()

        if self.is_global_zero:
            print_fields(self)

        self.trainer: Trainer = None

        self.state_items = []

        if 'elm_class' in self.tasks:
            self.elm_data_file = Path(self.elm_data_file).absolute()
            assert self.elm_data_file.exists(), f"ELM data file {self.elm_data_file} does not exist"
            self.global_shot_split: dict[str,np.ndarray] = {}
            self.rankwise_signal_window_metadata: dict[str,list] = {}
            self.elm_raw_signal_mean: float = None
            self.elm_raw_signal_stdev: float = None
            self.time_to_elm_quantiles: dict[float,float] = {}
            self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}
            self.state_items.extend([
                'global_shot_split',
            ])

        if 'conf_classifier' in self.tasks:
            self.confinement_data_file = Path(self.confinement_data_file).absolute()
            assert self.confinement_data_file.exists(), f"Confinement data file {self.confinement_data_file} does not exist"
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

        # FIR filter
        self.a_coeffs = self.b_coeffs = None
        self.set_fir_filter()

        # seed and RNG
        if self.seed is None:
            self.seed = np.random.default_rng().integers(0, 2**32-1)
            self.zprint(f"Using random seed: {self.seed}")
            self.save_hyperparameters("seed")
        self.rng = np.random.default_rng(self.seed)
        self.zprint(f"Using random number generator with seed {self.seed}")

    def prepare_data(self):
        self.zprint("\u2B1C Prepare data (rank 0 only)")
        if 'elm_class' in self.tasks:
            self.prepare_elm_data()
        if 'conf_classifier' in self.tasks and not self.global_confinement_shot_split:
            self.prepare_global_confinement_data()

    def prepare_elm_data(self):
        self.zprint("\u2B1C Prepare ELM data (rank 0 only)")
        t_tmp = time.time()
        # parse full dataset
        with h5py.File(self.elm_data_file, 'r') as root:
            # validate shots in data file
            datafile_shots = set([int(shot_key) for shot_key in root['shots']])
            datafile_shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(datafile_shots ^ datafile_shots_from_elms) == 0
            datafile_shots = list(datafile_shots)
            datafile_elms = [int(elm_key) for elm_key in root['elms']]
            self.zprint(f"  ELMs/shots in data file: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            # check for reloaded data state
            if self.global_shot_split:
                self.zprint("  Global shot split was reloaded")
            else:
                self.zprint("  Shuffling datafile shots")
                self.rng.shuffle(datafile_shots)
                self.zprint("  Splitting datafile shots into train/val/test")
                n_test_shots = int(self.fraction_test * len(datafile_shots))
                n_validation_shots = int(self.fraction_validation * len(datafile_shots))
                self.global_shot_split['test'], self.global_shot_split['validation'], self.global_shot_split['train'] = \
                    np.split(datafile_shots, [n_test_shots, n_test_shots+n_validation_shots])
            # global shot split
            for stage, shotlist in self.global_shot_split.items():
                if len(shotlist)<=7:
                    self.zprint(f"    {stage.upper()} shots: {shotlist}")
                else:
                    self.zprint(f"    {stage.upper()} shots: {shotlist[0:7]}")
            assert 'train' in self.global_shot_split and len(self.global_shot_split['train'])>0
            # save state dict
            self.save_state_dict()
            # prepare ELMs for stages
            if self.max_elms:
                n_elms = {
                    'train': int((1-self.fraction_validation-self.fraction_test) * self.max_elms),
                    'validation': int(self.fraction_validation * self.max_elms),
                    'test': int(self.fraction_test * self.max_elms),
                }
            # prepare data for stages
            for sub_stage in ['train','validation','test']:
                self.zprint("\u2B1C " + f"Prepare ELM data for stage {sub_stage.upper()} (rank 0 only)")
                # global ELMs for stage
                global_elms_for_stage = [
                    i_elm for i_elm in datafile_elms
                    if root['elms'][f"{i_elm:06d}"].attrs['shot'] in self.global_shot_split[sub_stage]
                ]
                if len(global_elms_for_stage) == 0:
                    continue
                self.rng.shuffle(global_elms_for_stage)
                if self.max_elms:
                    global_elms_for_stage = global_elms_for_stage[:n_elms[sub_stage]]
                self.zprint(f"  {sub_stage.upper()} ELM count: {len(global_elms_for_stage):,d}")
                if len(global_elms_for_stage) <= 7:
                    self.zprint(f"  {sub_stage.upper()} ELM IDs: {global_elms_for_stage}")
                else:
                    self.zprint(f"  {sub_stage.upper()} ELM IDs: {global_elms_for_stage[0:7]}")
                global_signal_window_metadata = []
                last_stat_elm_index: int = -1
                skipped_short_pre_elm_time: int = 0
                global_outliers: int = 0
                sw_count: int = 0
                stat_count: int = 0
                n_bins = 200
                signal_min = np.array(np.inf)
                signal_max = np.array(-np.inf)
                cummulative_hist = np.zeros(n_bins, dtype=int)
                # get signal window metadata for global ELMs in stage
                for i_elm, elm_index in enumerate(global_elms_for_stage):
                    if i_elm%100 == 0:
                        self.zprint(f"    Reading ELM event {i_elm:04d}/{len(global_elms_for_stage):04d}")
                    elm_event: h5py.Group = root['elms'][f"{elm_index:06d}"]
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
                    assert signals.shape[0] == 64 and signals.shape[1] == bes_time.size
                    while True:
                        i_window_start = i_window_stop - self.signal_window_size
                        if i_window_start < i_start: break
                        # remove signal windows with outliers in raw signals
                        if self.outlier_value:
                            signal_window = signals[..., i_window_start:i_window_stop]
                            assert signal_window.shape[-1] == self.signal_window_size
                            if np.abs(signal_window).max() > self.outlier_value:
                                i_window_stop -= self.signal_window_size // self.stride_factor
                                global_outliers += 1
                                continue
                        sw_count += 1
                        global_signal_window_metadata.append({
                            'elm_index': elm_index,
                            'shot': shot,
                            'i_t0': i_window_start,
                            'time_to_elm': bes_time[i_stop] - bes_time[i_window_stop]
                        })
                        if sw_count % 500 == 0:
                            stat_count +=1
                            if elm_index != last_stat_elm_index:
                                fsignals = (
                                    np.array(
                                        scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
                                        dtype=np.float32,
                                    ) if self.b_coeffs is not None else signals
                                )
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
                self.zprint(f"  Valid signal windows: {sw_count:,d}")
                self.zprint(f"  Skipped signal windows for outliers (threshold {self.outlier_value} V): {global_outliers:,d}")
                self.zprint(f"  Skipped ELMs for short pre-ELM time (threshold {self.min_pre_elm_time} ms): {skipped_short_pre_elm_time:,d}")

                bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
                mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
                stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
                exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
                self.zprint(f"  Signal stats (post-FIR, if used):  mean {mean:.3f}  stdev {stdev:.3f}  exkurt {exkurt:.3f}  min/max {signal_min:.3f}/{signal_max:.3f}")

                if sub_stage == 'train':
                    self.zprint(f"  Using {sub_stage.upper()} for standardizing mean and stdev")
                    self.elm_raw_signal_mean = mean.item()
                    self.elm_raw_signal_stdev = stdev.item()
                    self.zprint(f"  Standarizing signals with mean {self.elm_raw_signal_mean:.3f} and std {self.elm_raw_signal_stdev:.3f}")

                    quantiles = [0.5]
                    time_to_elm_labels = [sig_win['time_to_elm'] for sig_win in global_signal_window_metadata]
                    quantile_values = np.quantile(time_to_elm_labels, quantiles)
                    self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
                    self.zprint(f"  Time-to-ELM quantiles for binary labels:")
                    for q, qval in self.time_to_elm_quantiles.items():
                        self.zprint(f"    Quantile {q:.2f}: {qval:.1f} ms")

                if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
                    time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in global_signal_window_metadata])
                    time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
                    if self.contrastive_learning:
                        self.zprint(f"  Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                        for i in np.arange(len(global_signal_window_metadata)-1, -1, -1, dtype=int):
                            if (global_signal_window_metadata[i]['time_to_elm'] > time_to_elm_min) and \
                                (global_signal_window_metadata[i]['time_to_elm'] < time_to_elm_max):
                                global_signal_window_metadata.pop(i)
                    else:
                        self.zprint(f"  Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                        for i in np.arange(len(global_signal_window_metadata)-1, -1, -1, dtype=int):
                            if (global_signal_window_metadata[i]['time_to_elm'] < time_to_elm_min) or \
                                (global_signal_window_metadata[i]['time_to_elm'] > time_to_elm_max):
                                global_signal_window_metadata.pop(i)
                # balance signal windows across world_size
                remainder = len(global_signal_window_metadata) % self.trainer.world_size
                if remainder:
                    global_signal_window_metadata = global_signal_window_metadata[:-remainder]
                assert len(global_signal_window_metadata) % self.trainer.world_size == 0
                self.zprint(f"  Global signal window count (final): {len(global_signal_window_metadata):,d}")
                # split signal windows across ranks
                self.rankwise_signal_window_metadata[sub_stage] = np.array_split(
                    global_signal_window_metadata, 
                    self.trainer.world_size,
                )
        self.zprint(f"  ELM data prepare time: {time.time()-t_tmp:0.1f} s")

    def setup(self, stage: str):
        # called on all ranks after "prepare_data()"
        self.zprint("\u2B1C " + f"Setup stage {stage} (all ranks)")

        assert stage in ['fit', 'test', 'predict'], f"Invalid stage: {stage}"
        assert self.is_global_zero == self.trainer.is_global_zero

        sub_stages = ['train', 'validation'] if stage == 'fit' else [stage]

        if 'elm_class' in self.tasks:
            t_tmp = time.time()
            for sub_stage in sub_stages:
                self.setup_elm_data_for_rank(sub_stage)
            self.zprint(f"  ELM data setup time: {time.time()-t_tmp:0.1f} s")
        self.barrier()

        if 'conf_classifier' in self.tasks:
            t_tmp = time.time()
            self.zprint("**** Confinement data setup")
            for sub_stage in sub_stages:
                self._setup_confinement_data(sub_stage)
            self.zprint(f"  Confinement data setup time: {time.time()-t_tmp:.1f} s")
        self.barrier()

    def setup_elm_data_for_rank(self, sub_stage: str):
        self.rprint("  \u2B1C " + f"Setup ELM data for stage {sub_stage.upper()}")
        # broadcast
        self.rankwise_signal_window_metadata = self.broadcast(self.rankwise_signal_window_metadata)
        self.elm_raw_signal_mean = self.broadcast(self.elm_raw_signal_mean)
        self.elm_raw_signal_stdev = self.broadcast(self.elm_raw_signal_stdev)
        self.time_to_elm_quantiles = self.broadcast(self.time_to_elm_quantiles)
        # get rank-wise ELM signals
        sw_for_rank = list(self.rankwise_signal_window_metadata[sub_stage][self.trainer.global_rank])
        elms_for_rank = np.unique(np.array([item['elm_index'] for item in sw_for_rank],dtype=int))
        elms_to_signals: dict[int,torch.Tensor] = {}
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
                elms_to_signals[elm_index] = torch.from_numpy(
                    (signals - self.elm_raw_signal_mean) / self.elm_raw_signal_stdev
                )
        assert len(elms_to_signals) == len(elms_for_rank)
        signal_memory_size = sum([array.nbytes for array in elms_to_signals.values()])
        self.rprint(f"    Signal memory size: {signal_memory_size/(1024**3):.3f} GB")

        # rank-wise datasets
        if sub_stage in ['train', 'validation', 'test']:
            self.elm_datasets[sub_stage] = ELM_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                time_to_elm_quantiles=self.time_to_elm_quantiles,
                sw_list=sw_for_rank,
                elms_to_signals_dict=elms_to_signals,
            )
        elif sub_stage == 'predict':
            pass

    def get_elm_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.elm_datasets[stage],
            num_replicas=1,
            rank=0,
            shuffle=True if stage=='train' else False,
            seed=int(self.rng.integers(0, 2**32-1)),
            drop_last=True if stage=='train' else False,
        )
        batch_size: int = None
        if isinstance(self.batch_size, int):
            batch_size = self.batch_size
            pin_memory = True
        elif isinstance(self.batch_size, dict):
            pin_memory = False
            for key, value in reversed(self.batch_size.items()):
                if self.trainer.current_epoch >= key:
                    batch_size = value
                    break
        assert batch_size > 0
        return torch.utils.data.DataLoader(
            dataset=self.elm_datasets[stage],
            sampler=sampler,
            batch_size=batch_size//self.trainer.world_size,  # batch size per rank
            num_workers=self.num_workers,
            drop_last=True if stage=='train' else False,
            prefetch_factor=2 if self.num_workers else None,
            pin_memory=pin_memory,
            persistent_workers=True if self.num_workers else False,
        )

    def prepare_global_confinement_data(self):
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

    def _setup_confinement_data(self, stage: str):
        t_tmp = time.time()
        self.zprint(f"  {stage.upper()}")
        self.global_stage_to_events = self.broadcast(self.global_stage_to_events)
        global_stage_events = self.global_stage_to_events[stage]
        rankwise_stage_event_split: list = None
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
            self.rng.shuffle(packaged_valid_t0_indices)
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
        # self.zprint(f"    Batches per epoch: {len(packaged_valid_t0_indices)*self.trainer.world_size/self.batch_size:.1f}")
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

    def _conf_train_val_test_dataloaders(self, stage: str) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.DistributedSampler(
            dataset=self.confinement_datasets[stage],
            num_replicas=1,
            rank=0,
            shuffle=True if stage=='train' else False,
            seed=int(self.rng.integers(0, 2**32-1)),
            drop_last=True if stage=='train' else False,
        )
        return torch.utils.data.DataLoader(
            dataset=self.confinement_datasets[stage],
            sampler=sampler,
            batch_size=self.batch_size//self.trainer.world_size,  # batch size per rank
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True if stage=='train' else False,
        )

    def set_fir_filter(self):
        if self.fir_bp:
            assert len(self.fir_bp) == 2, "fir_bp must be a sequence of (f_low, f_high)"
        if self.fir_bp and (self.fir_bp[0] or self.fir_bp[1]):
            f_low, f_high = self.fir_bp
            self.zprint(f"Using FIR filter with f_low-f_high: {f_low}-{f_high} kHz")
            if f_low and f_high:
                pass_zero = 'bandpass'
                cutoff = [f_low, f_high]
            elif f_low:
                pass_zero = 'highpass'
                cutoff = f_low
            elif f_high:
                pass_zero = 'lowpass'
                cutoff = f_high
            self.b_coeffs = scipy.signal.firwin(
                numtaps=501,  # must be odd
                cutoff=cutoff,  # transition width in kHz
                pass_zero=pass_zero,
                fs=1e3,  # f_sample in kHz
            )
            self.a_coeffs = np.zeros_like(self.b_coeffs)
            self.a_coeffs[0] = 1
        else:
            self.zprint("Using raw BES signals; no FIR filter")

    def train_dataloader(self) -> CombinedLoader:
        loaders = {}
        if 'elm_class' in self.tasks:
            loaders['elm_class'] = self.get_elm_dataloaders('train')
        if 'conf_classifier' in self.tasks:
            loaders['conf_classifier'] = self._conf_train_val_test_dataloaders('train')
        combined_loader = CombinedLoader(
            iterables=loaders,
            mode='max_size_cycle',
        )
        _ = iter(combined_loader)
        return combined_loader

    def val_dataloader(self) -> CombinedLoader:
        loaders = {}
        if 'elm_class' in self.tasks:
            loaders['elm_class'] = self.get_elm_dataloaders('validation')
        if 'conf_classifier' in self.tasks:
            loaders['conf_classifier'] = self._conf_train_val_test_dataloaders('validation')
        combined_loader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def test_dataloader(self) -> CombinedLoader:
        loaders = {}
        if 'elm_class' in self.tasks:
            loaders['elm_class'] = self.get_elm_dataloaders('test')
        if 'conf_classifier' in self.tasks:
            loaders['conf_classifier'] = self._conf_train_val_test_dataloaders('test')
        combined_loader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def predict_dataloader(self) -> None:
        pass

    def load_state_dict(self, state: dict) -> None:
        self.zprint(f"  Loading state dict keys: {self.state_items}")
        for item in self.state_items:
            setattr(self, item, state[item])

    def save_state_dict(self):
        if self.is_global_zero:
            state_dict_file = Path(self.log_dir)/'state_dict.pt'
            state_dict_file.parent.mkdir(parents=True, exist_ok=True)
            self.zprint(f"  Saving state_dict: {state_dict_file}")
            state_dict = {item: getattr(self, item) for item in self.state_items}
            self.zprint(f"  State dict keys: {list(state_dict.keys())}")
            torch.save(state_dict, state_dict_file)

    def rprint(self, text: str = ''):
        self.barrier()
        super().rprint(text)
        self.barrier()

    def barrier(self):
        if self.world_size > 1:
            self.trainer.strategy.barrier()

    def broadcast(self, obj):
        if self.world_size > 1:
            obj = self.trainer.strategy.broadcast(obj)
            self.barrier()
        return obj


@dataclasses.dataclass(eq=False)
class ELM_TrainValTest_Dataset(torch.utils.data.Dataset):
    signal_window_size: int = 0
    sw_list: list = None # global signal window data mapping to dataset index
    elms_to_signals_dict: dict[int,torch.Tensor] = None # rank-wise signals (map to ELM indices)
    time_to_elm_quantiles: dict[float, float] = None

    def __post_init__(self):
        super().__init__()

    def __len__(self) -> int:
        return len(self.sw_list)
    
    def __getitem__(self, i: int) -> tuple:
        sw_metadata = self.sw_list[i]
        i_t0 = sw_metadata['i_t0']
        time_to_elm = sw_metadata['time_to_elm']
        elm_index = sw_metadata['elm_index']
        signals = self.elms_to_signals_dict[elm_index]
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
        experiment_name: str = 'experiment_default',
        trial_name: str = None,
        restart_trial_name: str = None,
        wandb_id: str = None,
        signal_window_size: int = 256,
        # model
        # elm_classifier: bool = True,
        # conf_classifier: bool = False,
        no_bias: bool = False,
        batch_norm: bool = True,
        dropout: float = 0.0,
        feature_model_layers: Sequence[dict[str, LightningModule]] = None,
        mlp_tasks: dict[str, Sequence] = None,
        # elm_mean_loss_factor = None,
        # conf_mean_loss_factor = None,
        # optimizer
        use_optimizer: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        deepest_layer_lr_factor: float = 0.1,
        lr_warmup_epochs: int = 10,
        lr_scheduler_patience: int = 100,
        monitor_metric = None,
        # transfer learning with backbone model
        backbone_model_path: str|Path = None,
        backbone_first_n_layers: int = None,
        backbone_initial_ratio_lr: float = 1e-3,
        backbone_unfreeze_at_epoch: int = 50,
        backbone_warmup_rate: float = 2.,
        # loggers
        log_freq: int = 100,
        use_wandb: bool = False,
        # callbacks
        early_stopping_patience: int = 300,
        # trainer
        max_epochs = 2,
        gradient_clip_val: float = 1,
        gradient_clip_algorithm: str = 'value',
        skip_train: bool = False,
        skip_data: bool = False,
        precision = None,
        # data
        elm_data_file: str|Path = None,
        confinement_data_file: str|Path = None,
        max_elms: int = None,
        batch_size: int|dict = 128,
        fraction_validation: float = 0.1,
        fraction_test: float = 0.0,
        num_workers: int = 2,
        time_to_elm_quantile_min: float = None,
        time_to_elm_quantile_max: float = None,
        contrastive_learning: bool = False,
        min_pre_elm_time: float = None,
        fir_bp: Sequence[float] = None,
        max_shots_per_class: int = None,
        max_confinement_event_length: int = None,
        seed: int = 42,
) -> dict:

    ### SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    world_rank = int(os.getenv("SLURM_PROCID", default=0))
    if world_size > 0:
        print(f"World rank {world_rank} of size {world_size} on {num_nodes} node(s)")
    if num_nodes > 1:
        local_rank = int(os.getenv("SLURM_LOCALID", default=0))
        node_rank = int(os.getenv("SLURM_NODEID", default=0))
        print(f"  Local rank {local_rank} on node {node_rank}")
    is_global_zero = (world_rank == 0)

    def zprint(text):
        if is_global_zero:
            print(text)

    ### model
    zprint("\u2B1C Creating model")
    lit_model = Model(
        # elm_classifier=elm_classifier,
        # conf_classifier=conf_classifier,
        signal_window_size=signal_window_size,
        # elm_loss_weight=elm_mean_loss_factor,
        # conf_loss_weight=conf_mean_loss_factor,
        no_bias=no_bias,
        batch_norm=batch_norm,
        feature_model_layers=feature_model_layers,
        mlp_tasks=mlp_tasks,
        dropout=dropout,
        # optimizer
        use_optimizer=use_optimizer,
        lr=lr,
        lr_scheduler_patience=lr_scheduler_patience,
        deepest_layer_lr_factor=deepest_layer_lr_factor,
        lr_warmup_epochs=lr_warmup_epochs,
        weight_decay=weight_decay,
        monitor_metric=monitor_metric,
        # transfer learning with backbone model
        backbone_model_path=backbone_model_path,
        backbone_first_n_layers=backbone_first_n_layers,
    )

    assert lit_model.world_size == world_size
    assert lit_model.world_rank == world_rank

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
            auto_insert_metric_name=False,
            filename='best-epoch-{epoch:04d}',
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode=metric_mode,
            patience=early_stopping_patience,
            log_rank_zero_only=True,
            check_finite=True,
        ),
    ]
    if backbone_model_path and backbone_first_n_layers:
        backbone_scheduler = BackboneFinetuning(
            lambda_func=lambda lr: lr * backbone_warmup_rate,
            unfreeze_backbone_at_epoch=backbone_unfreeze_at_epoch,
            backbone_initial_ratio_lr=backbone_initial_ratio_lr,
        )
        callbacks.append(backbone_scheduler)

    ### loggers
    zprint("\u2B1C Creating loggers")
    loggers = []
    experiment_dir = Path(experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if restart_trial_name:
        trial_name = restart_trial_name
    elif trial_name is None:
        datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        slurm_identifier = os.getenv('UNIQUE_IDENTIFIER', None)
        trial_name = f"r{slurm_identifier}_{datetime_str}" if slurm_identifier else f"r{datetime_str}"
    zprint(f"Trial name: {trial_name}")
    zprint(f"Trial path: {experiment_dir/trial_name}")
    tb_logger = TensorBoardLogger(
        save_dir=experiment_dir.parent, # parent directory of the experiment directory
        name=experiment_name,  # experiment directory name
        version=trial_name,  # trial directory name within the experiment directory
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    use_wandb = use_wandb and not skip_train and not skip_data
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
        wandb_id = wandb_logger.version
        zprint(f"W&B ID/version: {wandb_id}")
    else:
        wandb_id = None

    ### initialize trainer
    zprint("\u2B1C Creating Trainer")
    if precision is None:
        precision = '16-mixed' if torch.cuda.is_available() else 32
    reload_dataloaders_every_n_epochs = (
        0 if isinstance(batch_size, int) else
        list(batch_size.keys())[1]
    )
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
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
    )

    assert trainer.world_size == world_size
    assert trainer.global_rank == world_rank
    assert trainer.is_global_zero == is_global_zero
    assert trainer.log_dir == tb_logger.log_dir

    ckpt_path = (
        experiment_dir / restart_trial_name / 'checkpoints/last.ckpt'
        if restart_trial_name
        else None
    )

    ### data
    if not skip_data:
        zprint("\u2B1C Creating data module")
        if ckpt_path and False:  # do not reload datamodule when restarting.
            pass
            # assert ckpt_path.exists(), f"Checkpoint does not exist: {ckpt_path}"
            # print(f"Loading data from checkpoint: {ckpt_path}")
            # lit_datamodule = Data.load_from_checkpoint(checkpoint_path=ckpt_path)
            # state_dict_file = experiment_dir/restart_trial_name/'state_dict.pt'
            # assert state_dict_file.exists(), f"State dict file does not exist: {state_dict_file}"
            # print(f"Loading state_dict from: {state_dict_file}")
            # state_dict = torch.load(state_dict_file, weights_only=False)
            # lit_datamodule.load_state_dict(state_dict)
        else:
            if ckpt_path:
                assert seed is not None
            lit_datamodule = Data(
                signal_window_size=signal_window_size,
                log_dir=trainer.log_dir,
                elm_data_file=elm_data_file,
                confinement_data_file=confinement_data_file,
                # elm_classifier=lit_model.elm_classifier,
                # conf_classifier=lit_model.conf_classifier,
                tasks=lit_model.task_names,
                max_elms=max_elms,
                batch_size=batch_size,
                fraction_test=fraction_test,
                fraction_validation=fraction_validation,
                num_workers=num_workers,
                time_to_elm_quantile_min=time_to_elm_quantile_min,
                time_to_elm_quantile_max=time_to_elm_quantile_max,
                contrastive_learning=contrastive_learning,
                min_pre_elm_time=min_pre_elm_time,
                fir_bp=fir_bp,
                max_shots_per_class=max_shots_per_class,
                max_confinement_event_length=max_confinement_event_length,
                seed=seed,
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

    zprint(f"Trial name: {trial_name}")
    zprint(f"Trial path: {experiment_dir/trial_name}")

    if use_wandb:
        zprint(f"W&B ID/version: {wandb_id}")
        wandb.finish()

    return {
        'trial_name': trial_name,
        'trial_path': experiment_dir/trial_name,
        'wandb_id': wandb_id,
    }

if __name__=='__main__':
    feature_model_layers = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
    )
    mlp_tasks={
        # 'elm_class': [None,16,1],
        'conf_classifier': [None, 16,4],
    }
    main(
        elm_data_file=ml_data.small_data_100,
        confinement_data_file='/global/homes/d/drsmith/project-ml/data-archive/confinement_data.20240112.hdf5',
        feature_model_layers=feature_model_layers,
        mlp_tasks=mlp_tasks,
        max_elms=20,
        # skip_train=True,
    )