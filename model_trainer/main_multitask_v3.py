from pathlib import Path
import dataclasses
from datetime import datetime
from typing import OrderedDict, Sequence
import os
import time
import re
import pickle

import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import scipy.signal
import scipy.signal.windows
import scipy.special
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
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import \
    LearningRateMonitor, EarlyStopping, ModelCheckpoint, BackboneFinetuning
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.combined_loader import CombinedLoader

# import torchmetrics.classification
from torchmetrics.classification import \
    BinaryF1Score, BinaryPrecision, BinaryRecall, \
    MulticlassF1Score, MulticlassPrecision, MulticlassRecall

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
        self.num_nodes = int(os.getenv('SLURM_NNODES', default=1))
        self.slurm_local_rank = int(os.getenv("SLURM_LOCALID", default=0))
        self.is_global_zero = self.world_rank == 0

        if self.world_rank > 0:
            assert self.world_size > 1

    def zprint(self, text: str = ''):
        if self.is_global_zero:
            print(text)

    def rprint(self, text: str = ''):
        # if self.world_size > 1:
        print(f"{text}  (Rank {self.world_rank})")
        # else:
        #     print(f"{text}")


@dataclasses.dataclass(eq=False)
class Model(_Base_Class, LightningModule):
    lr: float = 1e-3
    deepest_layer_lr_factor: float = 1.
    lr_scheduler_patience: int = 100
    lr_scheduler_threshold: float = 1e-3
    lr_warmup_epochs: int = 0
    weight_decay: float = 1e-4
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
    unfreeze_uncertainty_epoch: int = -1
    backbone_model_path: str|Path = None
    backbone_first_n_layers: int = None
    elmwise_f1_interval: int = 25

    def __post_init__(self):

        # init superclasses
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()
        self.trainer: Trainer = None
        self.run_dir: Path = None
        self.local_device: str = f'cuda:{self.slurm_local_rank}'

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

        self.task_names = list(self.mlp_tasks.keys())
        for task_name in self.task_names:
            assert task_name in ['elm_class', 'conf_onehot'], f"Unknown task {task_name}"
        self.is_multitask = len(self.task_names) > 1

        # set input layer size
        for task_name in self.task_names:
            self.mlp_tasks[task_name][0] = self.feature_space_size
            if 'class' in task_name:
                assert self.mlp_tasks[task_name][-1] == 1
            elif 'onehot' in task_name:
                assert self.mlp_tasks[task_name][-1] > 1

        # create MLP configs
        self.task_configs = {}
        for task_name in self.task_names:
            self.task_configs[task_name] = {}
            self.task_configs[task_name]['layers'] = self.mlp_tasks[task_name]
            if 'class' in task_name:
                self.elm_wise_results = {}
                self.task_configs[task_name]['metrics'] = {
                    'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
                    'f1_score': BinaryF1Score().to(self.local_device),
                    'precision_score': BinaryPrecision().to(self.local_device),
                    'recall_score': BinaryRecall().to(self.local_device),
                    'mean_stat': torch.mean,
                    'std_stat': torch.std,
                }
                self.task_configs[task_name]['monitor_metric'] = 'f1_score/val'
            elif 'onehot' in task_name:
                self.task_configs[task_name]['metrics'] = {
                    'ce_loss': torch.nn.functional.cross_entropy,
                    'f1_score': MulticlassF1Score(4).to(self.local_device),
                    'precision_score': MulticlassPrecision(4).to(self.local_device),
                    'recall_score': MulticlassRecall(4).to(self.local_device),
                    'mean_stat': lambda t: torch.abs(torch.mean(t)), #torch.mean,
                    'std_stat': torch.std,
                }
                self.task_configs[task_name]['monitor_metric'] = 'f1_score/val'

        # make task sub-models
        self.task_metrics: dict[str, dict] = {}
        self.task_models: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.task_log_sigma: torch.nn.ParameterDict = torch.nn.ParameterDict()
        for task_name in self.task_names:
            task_dict = self.task_configs[task_name]
            self.zprint(f'Task sub-model: {task_name}')
            task_layers: tuple[int] = task_dict['layers']
            task_metrics: dict = task_dict['metrics']
            self.task_models[task_name] = self.make_mlp_classifier(mlp_layers=task_layers)
            self.task_metrics[task_name] = task_metrics.copy()
            self.task_log_sigma.update({task_name: torch.nn.Parameter(torch.tensor([-1.]))})
            if self.monitor_metric is None:
                # if not specified, use `monitor_metric` from first task
                self.monitor_metric = f"{task_name}/{task_dict['monitor_metric']}"
        self.save_hyperparameters({'monitor_metric': self.monitor_metric})
        self.task_log_sigma.requires_grad_ = True if self.unfreeze_uncertainty_epoch == -1 else False

        self.zprint(f"Total model parameters: {self.param_count(self):,d}")

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
            # if i_layer > 0 and self.batch_norm:
            if self.batch_norm:
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
            i_layer = n_feature_layers + i_mlp_layer
            # # batch norm
            # if self.batch_norm:
            #     layer_name = f"L{i_layer:02d}_BatchNorm"
            #     mlp_layer_dict[layer_name] = torch.nn.BatchNorm1d(mlp_layers[i_mlp_layer])
            #     self.zprint(f'  {layer_name} (regularization)')
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
        if not self.backbone_model_path or not self.backbone_first_n_layers:
            return
        self.backbone_model_path = next((Path(self.backbone_model_path) / 'checkpoints').glob('best-epoch-*.ckpt'))
        assert self.backbone_model_path.exists(), f"Transfer model not found: {self.backbone_model_path}"
        self_param_names = [param_name for param_name, _ in self.named_parameters()]
        source_model = torch.load(
            self.backbone_model_path,
            weights_only=False,
        )
        src_state_dict = source_model['state_dict']
        for param_name in list(src_state_dict.keys()):
            if param_name.startswith('backbone'):
                src_state_dict.pop(param_name)
                continue
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

    def configure_optimizers(self):
        self.zprint("\u2B1C " + f"Creating {self.use_optimizer.upper()} optimizer")
        self.zprint(f"  lr: {self.lr:.1e}")
        self.zprint(f"  lr for deepest layer: {self.deepest_layer_lr_factor * self.lr:.1e}")
        self.zprint(f"  Warmup epochs: {self.lr_warmup_epochs}")
        self.zprint(f"  Reduce on plateau threshold {self.lr_scheduler_threshold:.1e} and patience {self.lr_scheduler_patience:d}")
        self.zprint(f"  Unfreeze uncertainty epoch: {self.unfreeze_uncertainty_epoch:d}")

        params_weights = {'params': []}
        params_biases = {'params': []}
        params_batchnorms = {'params': []}
        params_sigmas = {'params': []}

        for pname, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'BatchNorm' in pname:
                params_batchnorms['params'].append(p)
            elif 'sigma' in pname:
                if self.unfreeze_uncertainty_epoch == -1:
                    p.requires_grad_(True)
                    params_sigmas['params'].append(p)
                else:
                    p.requires_grad_(False)
            elif 'bias' in pname:
                params_biases['params'].append(p)
            elif 'weight' in pname:
                params_weights['params'].append(p)
            else:
                raise ValueError(f"Unknown parameter name {pname}")
        # params_biases['weight_decay'] = 0.
        params_batchnorms['lr'] = self.lr / 100
        # params_batchnorms['weight_decay'] = 0.
        if params_sigmas['params']:
            params_sigmas['lr'] = self.lr / 100
            params_sigmas['weight_decay'] = 0.
        # else:
        #     params_sigmas = {}

        # params_weights = {
        #     'params': [param for param_name, param in self.named_parameters() if 'weight' in param_name and param.requires_grad],
        # }
        # params_biases = {
        #     'params': [param for param_name, param in self.named_parameters() if 'bias' in param_name and param.requires_grad],
        #     'weight_decay': 0.,
        # }
        # params_sigmas = {
        #     'params': [param for param_name, param in self.named_parameters() if 'sigma' in param_name and param.requires_grad],
        #     'lr': self.lr * 1e-2,
        #     'weight_decay': 0.,
        # }

        optim_kwargs = {
            'params': [params_weights, params_biases, params_batchnorms, params_sigmas],
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
        schedulers = [
            {'scheduler': plateau_scheduler, 'monitor': self.monitor_metric},
        ]
        if self.lr_warmup_epochs:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=0.05,
                total_iters=self.lr_warmup_epochs,
            )
            schedulers.append({'scheduler': warmup_scheduler})
        return ( [optimizer], schedulers, )

    # def make_parameter_list(self) -> list[dict]:
    #     parameter_list = []
    #     n_feature_layers = 0
    #     for layer_name, _ in self.named_modules():
    #         if not layer_name.endswith(('Conv','FC')):
    #             continue
    #         n_feature_layers += 1
    #     lrs_for_feature_layers = np.logspace(
    #         np.log10(self.lr * self.deepest_layer_lr_factor),
    #         np.log10(self.lr),
    #         n_feature_layers,
    #     )
    #     for param_name, params in self.named_parameters():
    #         param_dict = {}
    #         # no weight_decay for biases
    #         if 'bias' in param_name:
    #             param_dict['weight_decay'] = 0.
    #         if 'BatchNorm' in param_name:
    #             # default lr for BatchNorm
    #             param_dict['lr'] = self.lr
    #         elif 'task_log_sigma' in param_name:
    #             param_dict['lr'] = self.lr * 1e-2
    #         else:
    #             # lr for Conv and FC layers
    #             assert 'Conv' in param_name or 'FC' in param_name
    #             assert param_name.endswith(('weight','bias'))
    #             i_layer = int(re.findall(r'L(\d+)_', param_name)[0])
    #             if param_name.endswith('weight'):
    #                 param_dict['lr'] = lrs_for_feature_layers[i_layer]
    #             else:              
    #                 param_dict['lr'] = lrs_for_feature_layers[i_layer]/10
    #         optimizer_params = ''.join([f'  {key} {value:.1e}' for key, value in param_dict.items()])
    #         self.zprint(f"  {param_name}: {optimizer_params}")
    #         param_dict['params'] = params
    #         parameter_list.append(param_dict)

    #     return parameter_list

    def training_step(self, batch, batch_idx, dataloader_idx=None) -> torch.Tensor:
        if not (self.current_epoch or batch_idx or dataloader_idx):
            self.zprint("Begin training")
        # if batch_idx%500 == 0:
        #     self.zprint(f"    {self.current_epoch}, {batch_idx}, {dataloader_idx}")
        output = self.update_step(
            batch, 
            batch_idx, 
            stage='train',
            dataloader_idx=dataloader_idx,
        )
        return output

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

    def predict_step(self, batch, batch_idx, dataloader_idx) -> bool:
        assert dataloader_idx is not None
        if batch_idx == 0:
            self.rprint(f"  Predict for dataloader_idx {dataloader_idx}")
            assert dataloader_idx not in self.predict_outputs
            self.predict_outputs[dataloader_idx] = []
        signals, labels, times = batch
        features = self.feature_model(signals)
        datamodule: Data = self.trainer.datamodule
        task = datamodule.loader_tasks[dataloader_idx]
        task_outputs = self.task_models[task](features)
        prediction_outputs = {
            'outputs': task_outputs.numpy(force=True),
            'labels': labels.numpy(force=True),
            'times': times.numpy(force=True),
            'batch_idx': batch_idx,
        }
        # for result_key, result_value in task_outputs.items():
        #     prediction_outputs[result_key] = result_value.numpy(force=True)
        self.predict_outputs[dataloader_idx].append(prediction_outputs)
        return True

    def update_step(
            self, 
            batch: dict, 
            batch_idx = None, 
            dataloader_idx = None,
            stage: str = '', 
    ) -> torch.Tensor:
        sum_loss = torch.Tensor([0.])
        prod_loss = torch.Tensor([0.])
        sum_score = 0.
        model_outputs = self(batch)
        for task in model_outputs:
            task_outputs: torch.Tensor = model_outputs[task]
            metrics = self.task_metrics[task]
            if task == 'elm_class' and dataloader_idx in [None, 0]:
                labels: torch.Tensor = batch[task][1][0.5] if isinstance(batch, dict) else batch[1][0.5]
                elm_indices: torch.Tensor = batch[task][2] if isinstance(batch, dict) else batch[2]
                task_outputs: torch.Tensor = task_outputs.reshape_as(labels)
                if (self.current_epoch%self.elmwise_f1_interval==0 and stage in ['train','val']) or \
                    stage=='test':
                    for i in range(elm_indices.shape[0]):
                        elm_index = elm_indices[i].item()
                        if elm_index not in self.elm_wise_results:
                            self.elm_wise_results[elm_index] = {
                                'labels': [], 
                                'outputs': [], 
                                'stage': stage,
                            }
                        self.elm_wise_results[elm_index]['labels'].append(labels[i].item())
                        self.elm_wise_results[elm_index]['outputs'].append(task_outputs[i].item())
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs,
                            target=labels.type_as(task_outputs),
                        )
                        prod_loss = prod_loss * metric_value if prod_loss else metric_value
                        metric_weighted = (
                            metric_value / (torch.exp(self.task_log_sigma[task])) + self.task_log_sigma[task]
                            if self.is_multitask
                            else metric_value
                        )
                        sum_loss = sum_loss + metric_weighted if sum_loss else metric_weighted
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            preds=task_outputs,
                            target=labels,
                        )
                        if metric_name == 'f1_score':
                            sum_score = sum_score + metric_value if sum_score else metric_value
                    elif 'stat' in metric_name:
                        metric_value = metric_function(task_outputs)#.item()
                    self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True, add_dataloader_idx=False)
            elif task == 'conf_onehot' and dataloader_idx in [None, 1]:
                labels = (batch[task][1] if isinstance(batch, dict) else batch[1]).flatten()
                for metric_name, metric_function in metrics.items():
                    if 'loss' in metric_name:
                        metric_value = metric_function(
                            input=task_outputs,
                            target=labels,
                        )
                        prod_loss = prod_loss * metric_value if prod_loss else metric_value
                        metric_weighted = (
                            metric_value / (torch.exp(self.task_log_sigma[task])) + self.task_log_sigma[task]
                            if self.is_multitask
                            else metric_value
                        )
                        sum_loss = sum_loss + metric_weighted if sum_loss else metric_weighted
                    elif 'score' in metric_name:
                        metric_value = metric_function(
                            preds=task_outputs,
                            target=labels,
                        )
                        if metric_name == 'f1_score':
                            sum_score = sum_score + metric_value if sum_score else metric_value
                    elif 'stat' in metric_name:
                        metric_value = metric_function(task_outputs).item()
                    self.log(f"{task}/{metric_name}/{stage}", metric_value, sync_dist=True, add_dataloader_idx=False)

        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)
        self.log(f"prod_loss/{stage}", prod_loss, sync_dist=True)
        self.log(f"sum_score/{stage}", sum_score, sync_dist=True)
        return sum_loss

    def forward(
            self, 
            batch: dict|list, 
    ) -> dict[str, torch.Tensor]:
        results = {}
        for task in self.task_names:
            x = (
                batch[task][0]
                if isinstance(batch, dict) 
                else batch[0]
            )
            features = self.feature_model(x)
            results[task] = self.task_models[task](features)
        return results

    def setup(self, stage=None):  # fit, validate, test, or predict
        assert self.is_global_zero == self.trainer.is_global_zero
        if self.is_global_zero:
            assert self.global_rank == 0
        trainer: Trainer = self.trainer
        self.run_dir = Path(
            trainer.loggers[0].name,
            trainer.loggers[0].version,
        )
        trainer.strategy.barrier()
        if self.world_size > 0:
            print(f"World rank {self.world_rank} of size {self.world_size} on {self.num_nodes} node(s)")
        if self.num_nodes > 1:
            local_rank = int(os.getenv("SLURM_LOCALID", default=0))
            node_rank = int(os.getenv("SLURM_NODEID", default=0))
            print(f"  Local rank {local_rank} on node {node_rank}")
        trainer.strategy.barrier()

    def on_fit_start(self):
        self.t_fit_start = time.time()
        self.zprint(f"**** Fit start with max epochs {self.trainer.max_epochs} ****")
        # for param in self.task_log_sigma.parameters():
        #     param.requires_grad = False

    def on_validation_epoch_start(self):
        pass

    def on_test_start(self) -> None:
        self.elm_wise_results = {}
        return super().on_test_start()

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        self.s_train_epoch_start = self.global_step
        self.elm_wise_results = {}
        train_dataloader: CombinedLoader = self.trainer.train_dataloader
        for dl in train_dataloader.values():
            self.current_batch_size = dl.batch_size * self.trainer.world_size
            break
        lrs = np.array([
            param_group['lr']
            for optimizer in self.trainer.optimizers
            for param_group in optimizer.param_groups
        ])
        self.current_max_lr = lrs.max()
        self.current_min_lr = lrs.min()
        self.n_frozen_layers = 0
        for mod_name, module in self.named_modules():
            if 'task_log_sigma' in mod_name: continue
            for param_name, param in module.named_parameters(recurse=False):
                if 'task_log_sigma' in param_name: continue
                if not param.requires_grad:
                    # self.zprint(f"  Frozen {mod_name}.{param_name}")
                    self.n_frozen_layers += 1
                    break
        if self.is_multitask and self.current_epoch==self.unfreeze_uncertainty_epoch:
            pass
            # self.zprint(f"  Unfreezing task uncertainty parameters and adding to optimizer")
            # for p in self.task_log_sigma.parameters():
            #     p.requires_grad = True
            # params_sigmas = {
            #     'params': list(self.task_log_sigma.values()),
            #     'lr': self.lr / 10,
            #     'weight_decay': 0.,
            # }
            # optimizers = self.optimizers()
            # if isinstance(optimizers, list):
            #     for optimizer in optimizers:
            #         optimizer.add_param_group(params_sigmas)
            # else:
            #     optimizers.add_param_group(params_sigmas)

    def on_train_epoch_end(self):
        if self.is_multitask:
            for logger in self.loggers:
                logger.log_metrics(
                    metrics={
                        f'task_log_sigma/{task}': self.task_log_sigma[task].data.item()
                        for task in self.task_names
                    },
                    step=self.global_step,
                )
        if self.is_global_zero:
            epoch_time = time.time() - self.t_train_epoch_start
            global_time = time.time() - self.t_fit_start
            epoch_steps = self.global_step-self.s_train_epoch_start
            logged_metrics = self.trainer.logged_metrics
            line =  f"Ep {self.current_epoch:03d}"
            line += f"  bs {self.current_batch_size}"
            line += f" max lr {self.current_max_lr:.1e}"
            line += f" tr/val loss {logged_metrics['sum_loss/train']:.3f}/"
            sum_loss_val = (
                logged_metrics['sum_loss/val']
                if 'sum_loss/val' in logged_metrics
                else 
                logged_metrics['sum_loss/val/dataloader_idx_0']+logged_metrics['sum_loss/val/dataloader_idx_1']
            )
            line += f"{sum_loss_val:.3f}"
            line += f" ep/gl steps {epoch_steps:,d}/{self.global_step:,d}"
            line += f" ep/gl time (min): {epoch_time/60:.1f}/{global_time/60:.1f}" 
            if self.n_frozen_layers:
                line += f" n_frozen: {self.n_frozen_layers}" 
            print(line)
        self.save_elm_wise_f1_scores()

    def on_test_end(self) -> None:
        self.save_elm_wise_f1_scores()
        return super().on_test_end()

    def save_elm_wise_f1_scores(self):
        if not self.elm_wise_results:
            return
        # global ELM-wise F1 scores
        all_elm_wise_results = {}
        for i in range(self.trainer.world_size):
            all_elm_wise_results[i] = self.trainer.strategy.broadcast(self.elm_wise_results, src=i)
        if self.is_global_zero:
            elm_wise_results = {}
            for rank_results in all_elm_wise_results.values():
                # for elm_index in rank_results:
                #     assert elm_index not in elm_wise_results
                elm_wise_results.update(rank_results)
            elm_wise_f1_scores = {}
            for elm_index, elm_data in elm_wise_results.items():
                labels = np.array(elm_data['labels'], dtype=int)
                outputs = (np.array(elm_data['outputs']) >= 0.0).astype(int)
                f1_score = sklearn.metrics.f1_score(
                    y_true=labels, 
                    y_pred=outputs,
                    zero_division=0,
                )
                elm_wise_f1_scores[elm_index] = f1_score
            sorted_elm_scores_by_stage = {}
            for stage in ['train','val','test']:
                sorted_elm_indices = sorted(
                    [key for key in elm_wise_f1_scores.keys() if elm_wise_results[key]['stage']==stage],
                    key=lambda key: elm_wise_f1_scores[key],
                    reverse=True,
                )
                if sorted_elm_indices:
                    sorted_elm_scores_by_stage[stage] = {
                        elm_index: elm_wise_f1_scores[elm_index]
                        for elm_index in sorted_elm_indices
                    }
            if 'test' not in sorted_elm_scores_by_stage:
                pickle_file = self.run_dir / 'scores' / f'elm_wise_f1_scores_ep{self.current_epoch:04d}.pkl'
            else:
                pickle_file = self.run_dir / 'scores' / f'elm_wise_f1_scores_test.pkl'
            pickle_file.parent.mkdir(exist_ok=True)
            assert not pickle_file.exists()
            with open(pickle_file, 'wb') as f:
                self.zprint(f"  Saving ELM-wise F1 scores to {pickle_file}")
                pickle.dump(sorted_elm_scores_by_stage, f)

    @classmethod
    def read_elm_scores(cls, filename: str|Path, n: int = 3):
        filename = Path(filename)
        assert filename.exists()
        with open(filename, 'rb') as f:
            sorted_elm_scores_by_stage = pickle.load(f)
        plt.ioff()
        plt.figure(figsize=(4.25,3.5))
        for stage in sorted_elm_scores_by_stage:
            print(stage)
            sorted_keys = list(sorted_elm_scores_by_stage[stage].keys())
            for key in sorted_keys[:n]:
                print(f"  ELM ID {key}  F1 {sorted_elm_scores_by_stage[stage][key]:.3f}")
            for key in sorted_keys[-n:]:
                print(f"  ELM ID {key}  F1 {sorted_elm_scores_by_stage[stage][key]:.3f}")
            # plot histograms
            plt.clf()
            f1_scores = list(sorted_elm_scores_by_stage[stage].values())
            plt.hist(
                f1_scores,
                bins=20,
                range=(0,1),
            )
            plt.title(f'ELM-wise F1 score distribution | {stage.upper()}')
            plt.xlabel('F1 score')
            plt.ylabel('ELM count')
            plt.tight_layout()
            file_name = filename.parent / f'F1_dist_{stage}.png'
            plt.savefig(file_name, dpi=300)


    def on_fit_end(self) -> None:
        delt = time.time() - self.t_fit_start
        self.rprint(f"Fit time: {delt/60:0.1f} min")

    def on_predict_start(self) -> None:
        self.predict_outputs: dict[int,list] = {}
        self.zprint("  Predict start")
        self.trainer.strategy.barrier()

    def on_predict_end(self) -> None:
        self.trainer.strategy.barrier()
        if not self.predict_outputs:
            return
        plt.ioff()
        fig = plt.figure()
        n_boxcar = 8
        boxcar_window = scipy.signal.windows.boxcar(n_boxcar) / n_boxcar
        lambda_smooth = lambda x: np.convolve(x, boxcar_window, mode='valid')
        trainer: Trainer = self.trainer
        dataloaders: Sequence[torch.utils.data.DataLoader] = trainer.predict_dataloaders
        datamodule: Data = trainer.datamodule
        for i_dl, batch_list in self.predict_outputs.items():
            self.rprint(f"  Plotting predictions for dataloader {i_dl}")
            task = datamodule.loader_tasks[i_dl]
            times = np.concatenate([batch['times'] for batch in batch_list], axis=0).squeeze()
            labels = np.concatenate([batch['labels'] for batch in batch_list], axis=0).squeeze()
            outputs = np.concatenate([batch['outputs'] for batch in batch_list], axis=0).squeeze()
            dataset = dataloaders[i_dl].dataset
            i_row, i_col = 2, 3
            bes_channel = i_row*8 + i_col + 1
            bes_signal = dataset.signals[...,::25,i_row,i_col].numpy().squeeze()
            bes_time = dataset.time[::25]
            plt.clf()
            if task == 'elm_class':
                fig.set_size_inches((4.25,3.5))
                plt.plot(
                    bes_time, bes_signal/(3*np.std(bes_signal)), 
                    label=f'BES ch. {bes_channel} (filt/stand)', 
                    lw=0.5, c='k'
                )
                predictions = scipy.special.expit(outputs)
                pred_smoothed = lambda_smooth(predictions)
                plt.plot(times, labels, 
                         label='True label', lw=3)
                plt.plot(times, predictions, 
                         label='Predicted prob.', color='lightgreen', lw=1)
                plt.plot(times[n_boxcar-1:], pred_smoothed, 
                         label='Smoothed pred. prob.', color='green', lw=1.5)
                plt.ylim(-0.3,1.1)
                plt.axvline(dataset.t_stop, 
                            color='r', label='ELM onset', lw=3)
                plt.axvspan(bes_time[0], dataset.t_start, 
                            color='y', alpha=0.2, zorder=1)
                plt.axvspan(dataset.t_stop, bes_time[-1], 
                            color='y', alpha=0.2, zorder=1)
                plt.title(f'Shot {dataset.shot} | ELM ID {dataset.elm_index:04d}')
                plt.xlabel('Time (ms)')
                plt.ylabel(f'Scaled BES ch. {bes_channel} or probability')
                plt.legend(
                    loc='upper right',
                    labelspacing=0.2,
                    framealpha=0.8,
                    fontsize='small',
                )
                file_name = f'predict_elm_shot_{dataset.shot}_elmid_{dataset.elm_index:04d}.png'
            elif task == 'conf_onehot':
                fig.set_size_inches((4.25,6.25))
                axes = fig.subplots(nrows=2)
                # upper plot
                plt.sca(axes[0])
                plt.plot(
                    bes_time, bes_signal/(3*np.std(bes_signal)), 
                    label=f'BES ch. {bes_channel} (filt/stand)', 
                    lw=0.5, c='k'
                )
                label = labels[0]
                dt = times[1]-times[0]
                assert all(labels==label)
                ylim = plt.ylim()
                # plot ground truth
                plt.axvspan(
                    times[0]-dt, times[-1],
                    ymin=0.01, ymax=0.11,
                    facecolor=f'C{label}', 
                    edgecolor=None,
                    alpha=0.5,
                )
                plt.annotate(
                    text='Ground truth',
                    xy=(
                        times[0] + 0.02*(times[-1]-times[0]), 
                        ylim[0] + 0.03*(ylim[1]-ylim[0]),
                    ),
                    fontsize='large',
                )
                # plot smoothed predictions
                smoothed_outputs = np.ndarray((outputs.shape[0]-n_boxcar+1, 4), dtype=float)
                for i in range(4):
                    smoothed_outputs[:,i] = lambda_smooth(outputs[:,i])
                smoothed_argmax = smoothed_outputs.argmax(axis=1)
                n_outputs = smoothed_argmax.size
                smoothed_times = times[n_boxcar-1:]
                t0 = None
                for i in range(n_outputs):
                    previous_output = smoothed_argmax[i-1] if i-1>=0 else -1
                    output = smoothed_argmax[i]
                    next_output = smoothed_argmax[i+1] if i+1<=n_outputs-1 else -1
                    t0 = smoothed_times[i]-dt if output != previous_output else t0
                    if i==0: t0 = times[i]-dt
                    if output == next_output:
                        continue
                    tf = smoothed_times[i]
                    plt.axvspan(
                        t0, tf,
                        ymin=0.12, ymax=0.22,
                        facecolor=f'C{output}', 
                        edgecolor=None,
                        alpha=0.5,
                    )
                plt.annotate(
                    text='Smoothed predictions',
                    xy=(
                        times[0] + 0.02*(times[-1]-times[0]), 
                        ylim[0] + 0.14*(ylim[1]-ylim[0]),
                    ),
                    fontsize='large',
                )
                # plot raw predictions
                dt = times[1]-times[0]
                t0 = None
                outputs_argmax = outputs.argmax(axis=1)
                n_outputs = outputs_argmax.size
                for i in range(n_outputs):
                    previous_output = outputs_argmax[i-1] if i-1>=0 else -1
                    output = outputs_argmax[i]
                    next_output = outputs_argmax[i+1] if i+1<=n_outputs-1 else -1
                    t0 = times[i]-dt if output != previous_output else t0
                    if output == next_output:
                        continue
                    tf = times[i]
                    plt.axvspan(
                        t0, tf,
                        ymin=0.23, ymax=0.33,
                        facecolor=f'C{output}', 
                        edgecolor=None,
                        alpha=0.5,
                    )
                plt.annotate(
                    text='Raw predictions',
                    xy=(
                        times[0] + 0.02*(times[-1]-times[0]), 
                        ylim[0] + 0.25*(ylim[1]-ylim[0]),
                    ),
                    fontsize='large',
                )
                plt.title(f"Shot {dataset.shot} | event {dataset.event}")
                plt.xlabel('Time (ms)')
                plt.ylabel(f'Scaled BES ch. {bes_channel}')
                # lower plot
                plt.sca(axes[1])
                for i in range(4):
                    probs = scipy.special.expit(outputs[:,i])
                    sm_probs = scipy.special.expit(lambda_smooth(outputs[:,i]))
                    plt.plot(times, probs, c=f'C{i}', lw=0.6,)
                    plt.plot(times[n_boxcar-1:], sm_probs, c=f'C{i}', lw=2,)
                plt.xlim(axes[0].get_xlim())
                plt.ylabel('Probability predictions')
                plt.xlabel('Time (ms)')
                labs = ['L-mode','H-mode','QH-mode','WP QH-mode']
                rects = []
                for i in range(4):
                    rects.append(
                        patches.Rectangle(
                            (0,0),1,1,
                            facecolor=f'C{i}',
                            edgecolor=None,
                            alpha=0.5,
                        )
                    )
                for i, ax in enumerate(axes):
                    plt.sca(ax)
                    plt.legend(
                        rects, labs,
                        ncols=2,
                        fontsize='small',
                        loc='upper right' if i==0 else 'lower right',
                    )
                file_name = f'predict_conf_shot_{dataset.shot}_eventid_{dataset.event:04d}.png'
            plt.tight_layout()
            file_path = self.run_dir / 'figs' / file_name
            file_path.parent.mkdir(exist_ok=True)
            plt.savefig(file_path, dpi=300)
        trainer.strategy.barrier()

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True)

    @staticmethod
    def param_count(model: LightningModule) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclasses.dataclass(eq=False)
class Data(_Base_Class, LightningDataModule):
    # all data configuration
    log_dir: str|Path = None
    batch_size: int|dict = 128
    stride_factor: int = 8
    num_workers: int = 2
    fir_bp: Sequence[float] = (None, None)  # bandpass filter cut-on and cut-off frequencies in kHz
    outlier_value: float = 6
    fraction_validation: float = 0.1
    fraction_test: float = 0.0
    use_random_data: bool = False
    seed: int = None  # seed for ELM index shuffling; must be same across processes
    tasks: Sequence[str] = ('elm_class',)
    # ELM data configuration
    elm_data_file: str|Path = None
    max_elms: int = None
    time_to_elm_quantile_min: float = None
    time_to_elm_quantile_max: float = None
    contrastive_learning: bool = False
    min_pre_elm_time: float = None
    bad_elm_indices: Sequence[int] = ()
    # confinement data configuration
    confinement_data_file: str|Path = None
    bad_confinement_shots: Sequence[int] = ()
    force_validation_shots: Sequence[int] = ()
    force_test_shots: Sequence[int] = ()
    num_classes: int = 4
    metadata_bounds = {
        'r_avg': None,
        'z_avg': None,
        'delz_avg': None
    }
    n_rows: int = 8
    n_cols: int = 8
    max_confinement_event_length: int = None
    confinement_dataset_factor: float = None
    balance_confinement_data_with_elm_data: bool = False

    def __post_init__(self):
        super().__post_init__()
        super(_Base_Class, self).__init__()
        self.save_hyperparameters()

        if self.is_global_zero:
            print_fields(self)

        self.trainer: Trainer = None

        self.state_items = ['seed']

        self.allow_zero_length_dataloader_with_multiple_devices = True

        for task_name in self.tasks:
            assert task_name in ['elm_class', 'conf_onehot'], f"Unknown task {task_name}"

        self.signal_standardization_mean: float = None
        self.signal_standardization_stdev: float = None
        self.state_items.extend([
            'signal_standardization_mean',
            'signal_standardization_stdev',
        ])
        self.elm_sw_count_by_stage: dict[str, int] = {}
        self.conf_sw_count_by_stage: dict[str, int] = {}
        if 'elm_class' in self.tasks:
            self.elm_data_file = Path(self.elm_data_file).absolute()
            assert self.elm_data_file.exists(), f"ELM data file {self.elm_data_file} does not exist"
            self.global_elm_data_shot_split: dict[str,Sequence] = {}
            self.elm_signal_window_metadata: dict[str,Sequence[dict]] = {}
            self.state_items.extend([
                'global_elm_data_shot_split',
            ])
            self.time_to_elm_quantiles: dict[float,float] = {}
            self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}

        if 'conf_onehot' in self.tasks:
            self.confinement_data_file = Path(self.confinement_data_file).absolute()
            assert self.confinement_data_file.exists(), f"Confinement data file {self.confinement_data_file} does not exist"
            self.global_conf_data_shot_split: dict[str,Sequence] = {}
            self.state_items.extend([
                'global_conf_data_shot_split',
            ])
            self.stage_to_rank_to_event_mapping: dict[str,list[list]] = {}
            self.confinement_datasets: dict[str,torch.utils.data.Dataset] = {}

        for item in self.state_items:
            assert hasattr(self, item)
            if item != 'seed':
                assert not getattr(self, item)

        # FIR filter
        self.a_coeffs = self.b_coeffs = None
        self.set_fir_filter()

    def prepare_data(self):
        self.zprint("\u2B1C Prepare data (rank 0 only)")
        # seed and RNG
        if self.seed is None:
            self.seed = int(np.random.default_rng().integers(0, 2**32-1))
            self.zprint(f"  Randomly generated seed: {self.seed}")
            self.save_hyperparameters("seed")
        self.zprint(f"  Using random number generator with seed {self.seed}")
        self.prepare_rng = np.random.default_rng(self.seed)
        if 'elm_class' in self.tasks:
            self.prepare_elm_data()
        if 'conf_onehot' in self.tasks:
            self.prepare_confinement_data()
        self.save_state_dict()
        del self.prepare_rng

    def setup(self, stage: str):
        # called on all ranks after "prepare_data()"
        self.zprint("\u2B1C " + f"Setup stage {stage} (all ranks)")
        # self.rank_rng = np.random.default_rng()
        assert stage in ['fit', 'test', 'predict'], f"Invalid stage: {stage}"
        assert self.is_global_zero == self.trainer.is_global_zero
        assert self.world_size == self.trainer.world_size
        assert self.world_rank == self.trainer.global_rank
        sub_stages = ['train', 'validation'] if stage == 'fit' else [stage]
        if 'elm_class' in self.tasks:
            t_tmp = time.time()
            for sub_stage in sub_stages:
                self.setup_elm_data_for_rank(sub_stage)
            self.zprint(f"  ELM data setup time: {time.time()-t_tmp:0.1f} s")
            self.barrier()
        if 'conf_onehot' in self.tasks:
            t_tmp = time.time()
            for sub_stage in sub_stages:
                self.setup_confinement_data_for_rank(sub_stage)
            self.save_hyperparameters({
                'conf_sw_count_by_stage': self.conf_sw_count_by_stage,
            })
            self.zprint(f"  Confinement data setup time: {time.time()-t_tmp:.1f} s")
            self.barrier()
        self.zprint("\u2B1C Data setup summary")
        for sub_stage in sub_stages:
            self.zprint(f"  Stage {sub_stage}")
            for task in self.tasks:
                datasets = (
                    self.elm_datasets[sub_stage] 
                    if task == 'elm_class' 
                    else self.confinement_datasets[sub_stage]
                )
                n_samples = (
                    len(datasets)
                    if 'predict' not in sub_stage.lower()
                    else sum([len(ds) for ds in datasets])
                )
                batches = n_samples / (self.batch_size/self.world_size)
                out = f"    {task}  n_samples: {n_samples:,d}  batches/epoch: {batches:.1f}"
                if 'predict' in sub_stage.lower():
                    out += f"  datasets: {len(datasets)}"
                self.rprint(out)
                self.barrier()
        if stage == 'fit':
            self.save_state_dict()

    def prepare_elm_data(self):
        self.zprint("  \u2B1C Prepare ELM data (rank 0 only)")
        t_tmp = time.time()
        if self.elm_signal_window_metadata and all([bool(item) for item in self.elm_signal_window_metadata.values()]):
            self.zprint("    ELM signal window metadata was pre-loaded")
            return
        # parse full dataset
        with h5py.File(self.elm_data_file, 'r') as root:
            # validate shots in data file
            datafile_shots = set([int(shot_key) for shot_key in root['shots']])
            datafile_shots_from_elms = set([int(elm_group.attrs['shot']) for elm_group in root['elms'].values()])
            assert len(datafile_shots ^ datafile_shots_from_elms) == 0
            datafile_shots = list(datafile_shots)
            datafile_elms = [int(elm_key) for elm_key in root['elms']]
            self.zprint(f"    ELMs/shots in data file: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            if self.bad_elm_indices:
                self.save_hyperparameters({
                    'bad_elm_indices': self.bad_elm_indices,
                })
                self.zprint(f"    Excluding {len(self.bad_elm_indices):,d} bad ELM indices")
                datafile_elms = [
                    elm 
                    for elm in datafile_elms
                    if elm not in self.bad_elm_indices
                ]
                datafile_shots = list(set([int(root['elms'][f"{elm_id:06d}"].attrs['shot']) for elm_id in datafile_elms]))
                self.zprint(f"    ELMs/shots in data file with exclusions: {len(datafile_elms):,d} / {len(datafile_shots):,d}")
            # check for reloaded data state
            if self.global_elm_data_shot_split:
                self.zprint("    Global shot split was pre-loaded")
            else:
                self.zprint("    Shuffling datafile shots")
                # shuffle shots in database
                self.prepare_rng.shuffle(datafile_shots)
                self.zprint("    Splitting datafile shots into train/val/test")
                n_test_shots = int(self.fraction_test * len(datafile_shots))
                n_validation_shots = int(self.fraction_validation * len(datafile_shots))
                # split shots in database into train/val/test
                self.global_elm_data_shot_split['test'] = datafile_shots[:n_test_shots]
                self.global_elm_data_shot_split['validation'] = datafile_shots[n_test_shots:n_test_shots+n_validation_shots]
                self.global_elm_data_shot_split['train'] = datafile_shots[n_test_shots+n_validation_shots:]
                self.global_elm_data_shot_split['predict'] = self.global_elm_data_shot_split['test']
            # global shot split
            for stage, shotlist in self.global_elm_data_shot_split.items():
                self.zprint(f"      {stage.upper()} shots: {shotlist if len(shotlist)<=7 else shotlist[0:7]}")
            for stage in ['train','validation']:
                assert stage in self.global_elm_data_shot_split and len(self.global_elm_data_shot_split[stage])>0
            # prepare ELMs for stages
            if self.max_elms:
                n_elms = {
                    'train': int((1-self.fraction_validation-self.fraction_test) * self.max_elms),
                    'validation': int(self.fraction_validation * self.max_elms),
                    'test': int(self.fraction_test * self.max_elms),
                }
            # prepare data for stages
            for sub_stage in ['train','validation','test']:
                # global ELMs for stage
                if len(self.global_elm_data_shot_split[sub_stage]) == 0 \
                    or sub_stage in self.elm_signal_window_metadata:
                    continue
                self.zprint("    \u2B1C " + f"Prepare ELM data for {sub_stage.upper()} (rank 0 only)")
                global_elms_for_stage = [
                    i_elm for i_elm in datafile_elms
                    if root['elms'][f"{i_elm:06d}"].attrs['shot'] in self.global_elm_data_shot_split[sub_stage]
                ]
                # shuffles ELMs in stage
                self.prepare_rng.shuffle(global_elms_for_stage)
                # limit max ELMs in stage
                if self.max_elms:
                    global_elms_for_stage = global_elms_for_stage[:n_elms[sub_stage]]
                self.zprint(f"      ELM count: {len(global_elms_for_stage):,d}")
                if len(global_elms_for_stage) <= 7:
                    self.zprint(f"      ELM IDs: {global_elms_for_stage}")
                else:
                    self.zprint(f"      ELM IDs: {global_elms_for_stage[0:7]}")
                last_stat_elm_index: int = -1
                skipped_short_pre_elm_time: int = 0
                outliers: int = 0
                stat_count: int = 0
                n_bins = 200
                elm_signal_window_metadata: list[dict] = []
                signal_min = np.array(np.inf)
                signal_max = np.array(-np.inf)
                cummulative_hist = np.zeros(n_bins, dtype=int)
                # get signal window metadata for global ELMs in stage
                for i_elm, elm_index in enumerate(global_elms_for_stage):
                    if i_elm%100 == 0:
                        self.zprint(f"      Reading ELM event {i_elm:04d}/{len(global_elms_for_stage):04d}")
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
                                outliers += 1
                                continue
                        elm_signal_window_metadata.append({
                            'elm_index': elm_index,
                            'shot': shot,
                            'i_t0': i_window_start,
                            'time_to_elm': bes_time[i_stop] - bes_time[i_window_stop]
                        })
                        if len(elm_signal_window_metadata) % 500 == 0:
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
                self.zprint(f"      Skipped signal windows for outliers (threshold {self.outlier_value} V): {outliers:,d}")
                self.zprint(f"      Skipped ELMs for short pre-ELM time (threshold {self.min_pre_elm_time} ms): {skipped_short_pre_elm_time:,d}")
                self.zprint(f"      Global signal windows: {len(elm_signal_window_metadata):,d}")

                bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
                mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
                stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
                exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
                self.zprint(f"      Signal stats (post-FIR, if used):  mean {mean:.3f}  stdev {stdev:.3f}  exkurt {exkurt:.3f}  min/max {signal_min:.3f}/{signal_max:.3f}")

                # mean/stdev for standarization
                if sub_stage == 'train':
                    self.zprint('\u2B1C' + f" Using {sub_stage.upper()} for standardizing mean and stdev")
                    self.save_hyperparameters({
                        'elm_raw_signal_mean': mean.item(),
                        'elm_raw_signal_stdev': stdev.item(),
                    })
                    if not self.signal_standardization_mean:
                        self.signal_standardization_mean = mean.item()
                        self.signal_standardization_stdev = stdev.item()
                    self.zprint('\u2B1C' + f" Standarizing signals with mean {self.signal_standardization_mean:.3f} and std {self.signal_standardization_stdev:.3f}")
                    self.save_hyperparameters({
                        'standardization_mean': self.signal_standardization_mean,
                        'standardization_stdev': self.signal_standardization_stdev,
                    })

                    quantiles = [0.5]
                    time_to_elm_labels = [sig_win['time_to_elm'] for sig_win in elm_signal_window_metadata]
                    quantile_values = np.quantile(time_to_elm_labels, quantiles)
                    self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
                    self.zprint('\u2B1C' + f" Time-to-ELM quantiles for binary labels:")
                    for q, qval in self.time_to_elm_quantiles.items():
                        self.zprint('\u2B1C' + f"   Quantile {q:.2f}: {qval:.1f} ms")

                # set quantile limits
                if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
                    time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in self.elm_signal_window_metadata])
                    time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
                    # contrastive learning
                    if self.contrastive_learning:
                        self.zprint(f"  Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                        for i in np.arange(len(elm_signal_window_metadata)-1, -1, -1, dtype=int):
                            if (elm_signal_window_metadata[i]['time_to_elm'] > time_to_elm_min) and \
                                (elm_signal_window_metadata[i]['time_to_elm'] < time_to_elm_max):
                                elm_signal_window_metadata.pop(i)
                    else:
                        self.zprint(f"  Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                        for i in np.arange(len(elm_signal_window_metadata)-1, -1, -1, dtype=int):
                            if (elm_signal_window_metadata[i]['time_to_elm'] < time_to_elm_min) or \
                                (elm_signal_window_metadata[i]['time_to_elm'] > time_to_elm_max):
                                elm_signal_window_metadata.pop(i)

                # balance signal windows across world_size
                if self.world_size > 1:
                    remainder = len(elm_signal_window_metadata) % self.world_size
                    if remainder:
                        elm_signal_window_metadata = elm_signal_window_metadata[:-remainder]
                assert len(elm_signal_window_metadata) % self.world_size == 0
                self.zprint(f"  Rank-balanced global signal window count: {len(elm_signal_window_metadata):,d}")
                self.elm_signal_window_metadata[sub_stage] = elm_signal_window_metadata
                if sub_stage == 'test':
                    self.elm_signal_window_metadata['predict'] = self.elm_signal_window_metadata['test']
        self.elm_sw_count_by_stage = {stage: len(sigwins) for stage, sigwins in self.elm_signal_window_metadata.items()}
        self.save_hyperparameters({
            'elm_sw_count_by_stage': self.elm_sw_count_by_stage,
        })
        for state, count in self.elm_sw_count_by_stage.items():
            self.zprint(f"  {state.upper()}  n_sigwins: {count:,d}  batches/ep: {count / (self.batch_size):,.2f}")    
        self.zprint(f"  ELM data prepare time: {time.time()-t_tmp:0.1f} s")

    def setup_elm_data_for_rank(self, sub_stage: str):
        self.zprint("  \u2B1C " + f"Setup ELM data for stage {sub_stage.upper()}")
        if sub_stage=='predict' and 'predict' in self.elm_datasets and self.elm_datasets['predict']:
            self.zprint("    ELM `predict` dataset already setup")
            return
        # broadcast global ELM data specifications
        self.elm_signal_window_metadata = self.broadcast(self.elm_signal_window_metadata)
        # broadcast signal standardization
        self.signal_standardization_mean = self.broadcast(self.signal_standardization_mean)
        self.signal_standardization_stdev = self.broadcast(self.signal_standardization_stdev)
        # broadcast time-to-ELM quantiles for labeling
        self.time_to_elm_quantiles = self.broadcast(self.time_to_elm_quantiles)

        # get rank-wise ELM signals
        sw_for_stage: np.ndarray = np.array(self.elm_signal_window_metadata[sub_stage], dtype=object)
        indices_for_ranks: list[np.ndarray] = np.array_split(
            range(len(sw_for_stage)), 
            self.trainer.world_size
        )
        # sw_per_rank = len(sw_for_stage) // self.trainer.world_size
        # sw_for_rank = sw_for_stage[self.trainer.global_rank*sw_per_rank:(self.trainer.global_rank+1)*sw_per_rank]
        sw_for_rank: np.ndarray = sw_for_stage[indices_for_ranks[self.trainer.global_rank]]
        elms_for_rank = np.unique(np.array([item['elm_index'] for item in sw_for_rank],dtype=int))
        elms_to_signals_map: dict[int, torch.Tensor] = {}
        elms_to_time_map: dict[int, np.ndarray] = {}
        elms_to_shot_map: dict[int, int] = {}
        elms_to_interelm_map: dict[int, tuple[float,float]] = {}
        with h5py.File(self.elm_data_file) as root:
            for elm_index in elms_for_rank:
                elm_group: h5py.Group = root['elms'][f"{elm_index:06d}"]
                elms_to_shot_map[elm_index] = int(elm_group.attrs['shot'])
                bes_time = np.array(elm_group["bes_time"], dtype=np.float32)  # (<time>)
                signals = np.array(elm_group["bes_signals"], dtype=np.float32)  # (64, <time>)
                assert signals.shape[1] == bes_time.size
                t_start: float = elm_group.attrs['t_start']
                t_stop: float = elm_group.attrs['t_stop']
                elms_to_time_map[elm_index] = bes_time
                elms_to_interelm_map[elm_index] = (t_start, t_stop)
                if self.b_coeffs is not None:
                    signals = np.array(
                        scipy.signal.lfilter(x=signals, a=self.a_coeffs, b=self.b_coeffs),
                        dtype=np.float32,
                    )
                signals = np.transpose(signals).reshape(1, -1, 8, 8)  # reshape to (time, pol, rad)
                # FIR first (if used), then normalize
                elms_to_signals_map[elm_index] = torch.from_numpy(
                    (signals - self.signal_standardization_mean) / self.signal_standardization_stdev
                )
        assert len(elms_to_signals_map) == len(elms_for_rank)
        signal_memory_size = sum([array.nbytes for array in elms_to_signals_map.values()])
        self.rprint(f"    {sub_stage.upper()} n_sigwins: {len(sw_for_rank):,d}  batches/ep: {len(sw_for_rank) / (self.batch_size//self.trainer.world_size):,.2f}  mem: {signal_memory_size/(1024**3):.3f} GB")
        self.barrier()

        # rank-wise datasets
        if sub_stage in ['train', 'validation', 'test']:
            self.elm_datasets[sub_stage] = ELM_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                time_to_elm_quantiles=self.time_to_elm_quantiles,
                sw_list=sw_for_rank,
                elms_to_signals_dict=elms_to_signals_map,
            )
        if sub_stage in ['test', 'predict']:
            dataset_list: list[ELM_Predict_Dataset] = []
            for i_elm in elms_to_signals_map:
                dataset_list.append(
                    ELM_Predict_Dataset(
                        signal_window_size=self.signal_window_size,
                        time_to_elm_quantiles=self.time_to_elm_quantiles,
                        signals=elms_to_signals_map[i_elm],
                        time=elms_to_time_map[i_elm],
                        interelm_times=elms_to_interelm_map[i_elm],
                        elm_index=i_elm,
                        shot=elms_to_shot_map[i_elm],
                        stride_factor=self.stride_factor,
                    )
                )
            rankwise_dataset_count = [
                self.broadcast(len(dataset_list), src=i)
                for i in range(self.world_size)
            ]
            max_rankwise_dataset_count = max(rankwise_dataset_count)
            while len(dataset_list) < max_rankwise_dataset_count:
                dataset_list.append(ELM_Predict_Dataset())
            self.elm_datasets['predict'] = dataset_list

    def prepare_confinement_data(self):
        self.zprint("  \u2B1C Prepare confinement data (rank 0 only)")
        check_bounds = lambda value, bounds: bounds[0] <= value <= bounds[1] if bounds else True
        r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        missing_inboard = bad_inboard = 0
        # create global class to shot mapping and shot to events mapping
        with h5py.File(self.confinement_data_file) as root:
            global_shot_to_events: dict[int,list] = {}
            global_class_to_shots: list[list] = [[] for _ in range(self.num_classes)]
            global_class_duration: list[int] = [0] * self.num_classes
            global_class_event_count: list[int] = [0] * self.num_classes
            global_class_signal_window_count: list[int] = [0] * self.num_classes
            for shot_key in root:
                if self.bad_confinement_shots and int(shot_key) in self.bad_confinement_shots:
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
                    event: dict[str,np.ndarray] = root[shot_key][event_key]
                    if 'labels' not in event:
                        continue
                    assert event['labels'].size == event['signals'].shape[1]
                    assert event['labels'][0] == event['labels'][-1]
                    assert event['signals'].shape[0] == 64
                    assert event['labels'][0] < self.num_classes
                    class_label = event['labels'][0].item()
                    event_length: int = event['labels'].size
                    if event['labels'].size < self.signal_window_size:
                        continue
                    valid_t0 = np.zeros(event['labels'].size, dtype=int)
                    valid_t0[self.signal_window_size-1::self.signal_window_size//self.stride_factor] = 1
                    valid_t0_indices = np.arange(valid_t0.size, dtype=int)
                    valid_t0_indices = valid_t0_indices[valid_t0 == 1]
                    n_signal_windows = len(valid_t0_indices)
                    events.append({
                        'shot': int(shot_key),
                        'event': int(event_key),
                        'shot_event_key': f"{shot_key}/{event_key}",
                        'class_label': class_label,
                        'event_length': event_length,
                        'sw_count': n_signal_windows,
                    })
                    global_class_to_shots[class_label].append(int(shot_key))
                    global_class_duration[class_label] += event_length
                    global_class_event_count[class_label] += 1
                    global_class_signal_window_count[class_label] += n_signal_windows
                if not events:
                    continue
                global_shot_to_events[int(shot_key)] = events

        global_class_to_shots = [list(set(item)) for item in global_class_to_shots]
        # exclusions
        self.zprint(f"    Data file exclusions")
        self.zprint(f"      missing inboard shot exclusions: {missing_inboard}")
        self.zprint(f"      bad inboard shot exclusions: {bad_inboard}")
        self.zprint(f"      r_avg shot exclusions: {r_avg_exclusions}")
        self.zprint(f"      z_avg shot exclusions: {z_avg_exclusions}")
        self.zprint(f"      delz_avg shot exclusions: {delz_avg_exclusions}")
        # data summary
        self.zprint("    Data file summary (after exclusions)")
        self.zprint(f"      Shots: {len(global_shot_to_events)}  n_sig_win {sum(global_class_signal_window_count):,d}")
        for i in range(self.num_classes):
            line =  f"      Class {i}:  shots {len(global_class_to_shots[i])}  "
            line += f"events {global_class_event_count[i]}  "
            line += f"duration {global_class_duration[i]:,d} mu-s  "
            line += f"n_sig_win {global_class_signal_window_count[i]:,d}"
            self.zprint(line)

        # randomly split global shots among train/val/test
        stages = ['train','validation','test']
        if self. global_conf_data_shot_split:
            self.zprint("    Using pre-loaded train/val/test shot split")
        else:
            self.zprint("    Data file stage to shot mapping")
            self.global_conf_data_shot_split = {st:[] for st in stages}
            p_train = 1. - self.fraction_validation - self.fraction_test
            for shot in global_shot_to_events:
                choice = self.prepare_rng.choice(
                    3, p=(p_train, self.fraction_validation, self.fraction_test)
                )
                self.global_conf_data_shot_split[stages[choice]].append(shot)
            self.zprint(f"      Shuffling shots within stages")
            for stage in stages:
                self.prepare_rng.shuffle(self.global_conf_data_shot_split[stage])
        self.global_conf_data_shot_split['predict'] = self.global_conf_data_shot_split['test']
        # for stage in stages:
        for stage, shots in self.global_conf_data_shot_split.items():
            self.zprint(f"      {stage.capitalize()}  n_shots: {len(shots)}  shots: {shots if len(shots)<=6 else shots[:6]}")

        # create global stage to class to shot mapping
        self.zprint("    Data file stage to class to shot mapping")
        global_stage_to_class_to_shot_mapping: dict[str, dict[int, list]] = \
            {st:{i:[] for i in range(self.num_classes)} for st in stages}
        for st in stages:
            if len(self.global_conf_data_shot_split[st])==0: continue
            self.zprint(f"      {st.capitalize()}")
            for i in range(self.num_classes):
                for shot in self.global_conf_data_shot_split[st]:
                    if shot in global_class_to_shots[i]:
                        global_stage_to_class_to_shot_mapping[st][i].append(shot)
                shots = global_stage_to_class_to_shot_mapping[st][i]
                self.zprint(f"        Class {i}  n_shots: {len(shots)}  shots: {shots if len(shots)<=6 else shots[:6]}")

        # if needed, trim stage and class shotlists
        stage_to_class_to_shot_mapping: dict[str, dict[int, list]] = \
            {st: {i:[] for i in range(self.num_classes)} for st in stages}
        if self.confinement_dataset_factor:
            self.zprint("    Training data stage to class to shot mapping")
            for st in global_stage_to_class_to_shot_mapping:
                if len(self.global_conf_data_shot_split[st])==0: continue
                self.zprint(f"      {st.capitalize()}")
                for i in range(self.num_classes):
                    n_shots = int(self.confinement_dataset_factor *
                                  len(global_stage_to_class_to_shot_mapping[st][i]))
                    if n_shots==0: n_shots = 1
                    stage_to_class_to_shot_mapping[st][i].extend(
                        global_stage_to_class_to_shot_mapping[st][i][:n_shots]
                    )
                    shots = stage_to_class_to_shot_mapping[st][i]
                    self.zprint(f"        Class {i}  n_shots: {len(shots)}  shots: {shots if len(shots)<=6 else shots[:6]}")
        else:
            stage_to_class_to_shot_mapping = global_stage_to_class_to_shot_mapping.copy()

        # stage to events mapping
        self.zprint("    Training data stage to class to event mapping")
        stage_to_event_mapping: dict[str,list] = {}
        for st in stage_to_class_to_shot_mapping:
            stage_to_event_mapping[st] = []
            for i in stage_to_class_to_shot_mapping[st]:
                shots = stage_to_class_to_shot_mapping[st][i]
                events = [event for shot in shots for event in global_shot_to_events[shot]]
                stage_to_event_mapping[st].extend(events)
            if len(stage_to_event_mapping[st]) == 0:
                continue
            class_to_shots = [[] for _ in range(self.num_classes)]
            class_to_events = [0] * self.num_classes
            class_to_duration: list[int] = [0] * self.num_classes
            class_to_sigwins: list[int] = [0] * self.num_classes
            for event in stage_to_event_mapping[st]:
                class_to_shots[event['class_label']].append(event['shot'])
                class_to_events[event['class_label']] += 1
                class_to_duration[event['class_label']] += int(event['event_length'])
                class_to_sigwins[event['class_label']] += int(event['sw_count'])
            class_to_shots = [list(set(l)) for l in class_to_shots]
            self.zprint(f"      {st.capitalize()}  Total sig wins: {sum(class_to_sigwins):,d}")
            for i in range(self.num_classes):
                line =  f"        Class {i}:  shots {len(class_to_shots[i])}  "
                line += f"events {class_to_events[i]}  "
                line += f"duration {class_to_duration[i]:,d}  "
                line += f"n_sig_win {class_to_sigwins[i]:,d}  "
                # line += f"n_batches {class_to_sigwins[i]//self.batch_size:,d}"
                self.zprint(line)

        # split events among ranks for each stage
        self.zprint(f"    Approx. rank-wise signal window counts")
        for sub_stage in stages:
            # sort events longest to shortest for stage
            stage_to_event_mapping[sub_stage] = sorted(
                stage_to_event_mapping[sub_stage],
                key=lambda e: e['sw_count'],
                reverse=True,
            )
            self.stage_to_rank_to_event_mapping[sub_stage] = [[] for _ in range(self.world_size)]
            # assign events to ranks based upon the shortest cumulative time in each rank
            rankwise_total_sw_counts = np.zeros(self.world_size, dtype=int)
            for event in stage_to_event_mapping[sub_stage]:
                argmin = np.argmin(rankwise_total_sw_counts)
                self.stage_to_rank_to_event_mapping[sub_stage][argmin].append(event)
                rankwise_total_sw_counts[argmin] += event['sw_count']
            with np.printoptions(formatter={'int': lambda s: f'{s:,d}'}):
                self.zprint(f"      {sub_stage.capitalize()}: {rankwise_total_sw_counts}")
            # sort events shortest to longest for each rank
            for i_rank in range(self.world_size):
                self.stage_to_rank_to_event_mapping[sub_stage][i_rank] = sorted(
                    self.stage_to_rank_to_event_mapping[sub_stage][i_rank],
                    key=lambda e: e['sw_count'],
                )
        self.stage_to_rank_to_event_mapping['predict'] = self.stage_to_rank_to_event_mapping['test']


        # Forced shots  
        # forced_test_shots_data = {}
        # forced_validation_shots_data = {}
        # if self.force_test_shots:
        #     for shot_number in self.force_test_shots:
        #         forced_test_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        # if self.force_validation_shots:
        #     for shot_number in self.force_validation_shots:
        #         forced_validation_shots_data[shot_number] = global_shot_to_events.pop(shot_number)
        # limit shots per class
        # if self.max_shots_per_class:
        #     self.zprint(f"    Limiting classes to {self.max_shots_per_class} shots (approx)")
        #     for i in range(self.num_classes):
        #         if len(global_class_to_shots[i]) > self.max_shots_per_class:
        #             global_class_to_shots[i] = global_class_to_shots[i][:self.max_shots_per_class]
            # for i_class in range(len(global_class_to_shots)-1, -1, -1):
            #     class_shots = global_class_to_shots[i_class]
            #     if len(class_shots) > self.max_shots_per_class:
            #         # down-select shots for each class
            #         global_class_to_shots[i_class] = self.rng.choice(
            #             a=class_shots, 
            #             size=self.max_shots_per_class, 
            #             replace=False,
            #         ).tolist()
            # updated_shot_list = set([shot for shots in global_class_to_shots for shot in shots])
            # global_shot_to_events = {shot: global_shot_to_events[shot] for shot in updated_shot_list}
        # redo shot/event data
        # with h5py.File(self.confinement_data_file) as root:
        #     global_class_to_shots: list[list] = [[] for _ in range(self.num_classes)]
        #     global_class_duration: list[int] = [0] * self.num_classes
        #     global_class_event_count: list[int] = [0] * self.num_classes
        #     global_class_signal_window_count: list[int] = [0] * self.num_classes
        #     for shot in global_shot_to_events:
        #         shot_key: str = f"{shot}"
        #         events: list[dict] = []
        #         for event_key in root[shot_key]:
        #             event = root[shot_key][event_key]
        #             class_label = event['labels'][0].item()
        #             event_length: int = event['labels'].size
        #             if event_length < self.signal_window_size:
        #                 continue
        #             if self.max_confinement_event_length:
        #                 event_length = min(event_length, self.max_confinement_event_length)
        #             valid_t0 = np.zeros(event_length, dtype=int)
        #             valid_t0[self.signal_window_size-1::self.signal_window_size//self.stride_factor] = 1
        #             valid_t0_indices = np.arange(valid_t0.size, dtype=int)
        #             valid_t0_indices = valid_t0_indices[valid_t0 == 1]
        #             n_signal_windows = len(valid_t0_indices)
        #             events.append({
        #                 'shot': int(shot_key),
        #                 'event': int(event_key),
        #                 'shot_event_key': f"{shot_key}/{event_key}",
        #                 'class_label': class_label,
        #                 'event_length': event_length,
        #                 'sw_count': n_signal_windows,
        #             })
        #             global_class_to_shots[class_label].append(int(shot_key))
        #             global_class_duration[class_label] += event_length
        #             global_class_event_count[class_label] += 1
        #             global_class_signal_window_count[class_label] += n_signal_windows
        # global_class_to_shots = [list(set(item)) for item in global_class_to_shots]
        # # data read
        # self.zprint("    Restricted data summary")
        # self.zprint(f"      Shots: {len(global_shot_to_events)}")
        # for i in range(self.num_classes):
        #     self.zprint(f"        Class {i}: shots {len(global_class_to_shots[i])} events {global_class_event_count[i]} duration {global_class_duration[i]:,d} n_sig_win {global_class_signal_window_count[i]:,d}")

            # if forced_test_shots_data or forced_validation_shots_data:
            #     global_shot_to_events.update(forced_validation_shots_data)
            #     global_shot_to_events.update(forced_test_shots_data)
            #     self.global_confinement_shot_split['validation'].extend(list(forced_validation_shots_data.keys()))
            #     self.global_confinement_shot_split['test'].extend(list(forced_test_shots_data.keys()))

            # self.global_stage_to_shot['predict'] = self.global_stage_to_shot['test']

    def setup_confinement_data_for_rank(self, sub_stage: str):
        self.zprint("  \u2B1C " + f"Setup confinement data for stage {sub_stage.capitalize()}")
        if sub_stage=='predict' and 'predict' in self.confinement_datasets and self.confinement_datasets['predict']:
            self.zprint("    Confinement `predict` dataset already setup")
            return
        t_tmp = time.time()
        # TODO move rankwise_stage_event_split to prepare_conf_data()
        self.stage_to_rank_to_event_mapping = self.broadcast(self.stage_to_rank_to_event_mapping)
        self.elm_sw_count_by_stage = self.broadcast(self.elm_sw_count_by_stage)

        # package data for rank
        rankwise_events = self.stage_to_rank_to_event_mapping[sub_stage][self.world_rank]
        n_events = len(rankwise_events)
        # pre-allocate signal array
        time_count = 0
        if self.max_confinement_event_length:
            self.zprint(f"    Max confinement event length: {self.max_confinement_event_length}")
        for event in rankwise_events:
            if self.max_confinement_event_length and event['event_length']>= self.max_confinement_event_length:
                time_count += self.max_confinement_event_length
            else:
                time_count += event['event_length']
        packaged_signals = np.empty((time_count, self.n_rows, self.n_cols), dtype=np.float32)
        rankwise_events_2 = []
        start_index = 0
        outlier_count = 0
        count_of_valid_t0_indices: int = 0
        shotevent_to_signals_map: dict[str, np.ndarray] = {}
        shotevent_to_time_map: dict[str, np.ndarray] = {}
        shotevent_to_labels_map: dict[str, np.ndarray] = {}
        shotevent_to_shot_and_event_map: dict[str, tuple[int,int]] = {}
        with h5py.File(self.confinement_data_file, 'r') as root:
            for i, event_data in enumerate(rankwise_events):
                if n_events >= 10 and i % (n_events//10) == 0:
                    self.zprint(f"      Reading event {i:04d}/{n_events:04d}")
                shot = event_data['shot']
                event = event_data['event']
                shotevent_label = f"{shot}_{event}"
                class_label = event_data['class_label']
                event_group = root[str(shot)][str(event)]
                labels = np.array(event_group["labels"], dtype=int)
                assert labels[0] == class_label and labels[-1] == class_label
                if self.max_confinement_event_length and labels.size>self.max_confinement_event_length:
                    max_length = self.max_confinement_event_length
                    labels = labels[:max_length]
                else:
                    max_length = len(labels)
                shotevent_to_labels_map[shotevent_label] = labels
                signals = np.array(event_group["signals"][:, :max_length], dtype=np.float32)
                bes_time = np.array(event_group["time"][:max_length], dtype=np.float32)  # (<time>)
                shotevent_to_time_map[shotevent_label] = bes_time
                shotevent_to_shot_and_event_map[shotevent_label] = (int(shot), int(event))
                assert signals.shape[0] == 64
                assert labels.size == signals.shape[1]
                signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                valid_t0 = np.zeros(labels.size, dtype=int)
                valid_t0[self.signal_window_size-1::self.signal_window_size//self.stride_factor] = 1
                valid_t0_indices = np.arange(valid_t0.size, dtype=int)
                valid_t0_indices = valid_t0_indices[valid_t0 == 1]
                for i in valid_t0_indices:
                    assert i - self.signal_window_size + 1 >= 0  # start slice test
                    assert i+1 <= valid_t0.size  # end slice test
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
                shotevent_to_signals_map[shotevent_label] = signals
                packaged_signals[start_index:start_index + signals.shape[0], ...] = signals
                start_index += signals.shape[0]
                event_2 = event_data.copy()
                event_2['labels'] = labels
                event_2['valid_t0'] = valid_t0
                rankwise_events_2.append(event_2)
                count_of_valid_t0_indices += np.sum(valid_t0)
                # if self.elm_sw_count_by_stage:
                #     if count_of_valid_t0_indices > 2*self.elm_sw_count_by_stage[sub_stage]//self.world_size:
                #         self.zprint(f"    Stopping conf data read early to match ELM data size")
                #         break

        self.zprint(f"    Outlier count: {outlier_count}")
        packaged_signals = packaged_signals[:start_index, ...]
        self.zprint(f"    Signal memory size: {packaged_signals.nbytes/(1024**3):.4f} GB")
        self.barrier()
        # assert start_index == packaged_signals.shape[0]
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
        assert len(packaged_valid_t0_indices)
        for i in packaged_valid_t0_indices:
            assert i - self.signal_window_size + 1 >= 0  # start slice
            assert i+1 <= packaged_valid_t0.size  # end slice

        # self.zprint(f"    Shuffling valid t0 indices")
        # self.rank_rng.shuffle(packaged_valid_t0_indices)

        # match valid t0 indices count across ranks
        if self.trainer.world_size > 1 and sub_stage in ['train', 'validation']:
            # count_valid_t0_indices = len(packaged_valid_t0_indices)
            # self.rprint(f"    Valid t0 indices (unbalanced across ranks): {count_valid_t0_indices:,d}")
            all_rank_count_valid_indices: list = [None for _ in range(self.trainer.world_size)]
            for i in range(self.trainer.world_size):
                all_rank_count_valid_indices[i] = self.broadcast(len(packaged_valid_t0_indices), src=i)
            length_limit = min(all_rank_count_valid_indices)
            packaged_valid_t0_indices = packaged_valid_t0_indices[:length_limit]
        self.rprint(f"    Valid t0 indices: {len(packaged_valid_t0_indices):,d}")

        # balance with ELM dataset, if applicable
        if self.elm_sw_count_by_stage and \
                self.balance_confinement_data_with_elm_data and \
                sub_stage in ['train', 'validation']:
            elm_stage_sw_count = self.elm_sw_count_by_stage[sub_stage]
            elm_stage_rank_sw_count = elm_stage_sw_count // self.world_size
            if len(packaged_valid_t0_indices) > elm_stage_rank_sw_count:
                self.zprint(f"    Balancing confinement data with ELM data")
                packaged_valid_t0_indices = packaged_valid_t0_indices[:elm_stage_rank_sw_count]
            self.zprint(f"    Final valid t0 indices for each rank: {len(packaged_valid_t0_indices):,d}")

        all_rank_count_valid_indices: list = [None for _ in range(self.trainer.world_size)]
        for i in range(self.trainer.world_size):
            all_rank_count_valid_indices[i] = self.broadcast(len(packaged_valid_t0_indices), src=i)
        self.conf_sw_count_by_stage[sub_stage] = sum(all_rank_count_valid_indices)

        # stats
        confinement_raw_signal_mean = None
        confinement_raw_signal_stdev = None
        if self.is_global_zero:
            signal_min = np.inf
            signal_max = -np.inf
            n_bins = 200
            cummulative_hist = np.zeros(n_bins, dtype=int)
            stat_interval = max(1, packaged_valid_t0_indices.size//int(1e3))
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
            if sub_stage == 'train' or not (confinement_raw_signal_mean and confinement_raw_signal_stdev):
                confinement_raw_signal_mean = mean.astype(np.float32)
                confinement_raw_signal_stdev = stdev.astype(np.float32)
                self.save_hyperparameters({
                    'confinement_raw_signal_mean': confinement_raw_signal_mean.item(),
                    'confinement_raw_signal_stdev': confinement_raw_signal_stdev.item(),
                })

        confinement_raw_signal_mean = self.broadcast(confinement_raw_signal_mean)
        confinement_raw_signal_stdev = self.broadcast(confinement_raw_signal_stdev)
        if self.signal_standardization_mean and self.signal_standardization_stdev:
            self.zprint(f"    Using existing mean/stdev for standardization")
        else:
            self.zprint(f"    Using confinement data mean/stdev for standardization")
            self.signal_standardization_mean = confinement_raw_signal_mean
            self.signal_standardization_stdev = confinement_raw_signal_stdev
            self.save_hyperparameters({
                'standardization_mean': self.signal_standardization_mean,
                'standardization_stdev': self.signal_standardization_stdev,
            })
        self.signal_standardization_mean = self.broadcast(self.signal_standardization_mean)
        self.signal_standardization_stdev = self.broadcast(self.signal_standardization_stdev)
        self.zprint(f"    Standarizing signals with mean {self.signal_standardization_mean:.3f} and std {self.signal_standardization_stdev:.3f}")
        packaged_signals = (packaged_signals - self.signal_standardization_mean) / self.signal_standardization_stdev

        self.zprint(f"    {sub_stage.upper()} data time: {time.time()-t_tmp:.1f} s")

        if sub_stage in ['train', 'validation', 'test']:
            self.confinement_datasets[sub_stage] = Confinement_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                signals=packaged_signals,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                window_start_indices=packaged_window_start,
                shot_event_keys=packaged_shot_event_key,
            )
        if sub_stage in ['test', 'predict']:
            dataset_list: list[Confinement_Predict_Dataset] = []
            for shotevent in shotevent_to_signals_map:
                dataset = Confinement_Predict_Dataset(
                    signal_window_size=self.signal_window_size,
                    signals=shotevent_to_signals_map[shotevent],
                    labels=shotevent_to_labels_map[shotevent],
                    time=shotevent_to_time_map[shotevent],
                    shot_event_keys=shotevent_to_shot_and_event_map[shotevent],
                    stride_factor=self.stride_factor,
                )
                dataset_list.append(dataset)
            rankwise_dataset_count = [
                self.broadcast(len(dataset_list), src=i)
                for i in range(self.world_size)
            ]
            max_rankwise_dataset_count = max(rankwise_dataset_count)
            while len(dataset_list) < max_rankwise_dataset_count:
                dataset_list.append(Confinement_Predict_Dataset())
            self.confinement_datasets['predict'] = dataset_list

    def get_dataloader_from_dataset(
            self, 
            stage: str, 
            dataset: torch.utils.data.Dataset,
    ) -> torch.utils.data.DataLoader:
        sampler = torch.utils.data.DistributedSampler(
            dataset=dataset,
            num_replicas=1,
            rank=0,
            shuffle=True if stage=='train' else False,
            seed=int(np.random.default_rng().integers(0, 2**32-1)),
            # drop_last=True if stage=='train' else False,
        )
        batch_size: int = None
        if isinstance(self.batch_size, int):
            batch_size = self.batch_size
        elif isinstance(self.batch_size, dict):
            for key, value in reversed(self.batch_size.items()):
                if self.trainer.current_epoch >= key:
                    batch_size = value
                    break
        assert batch_size > 0
        num_workers = 0 if stage=='predict' else self.num_workers
        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size//self.trainer.world_size,  # batch size per rank
            num_workers=num_workers,
            # drop_last=True if stage=='train' else False,
            prefetch_factor=2 if num_workers else None,
            pin_memory=True if isinstance(self.batch_size, int) else False,
            persistent_workers=True if num_workers else False,
        )

    # def get_confinement_dataloaders(
    #         self, 
    #         stage: str,
    #         dataset: torch.utils.data.Dataset,
    # ) -> torch.utils.data.DataLoader:
    #     sampler = torch.utils.data.DistributedSampler(
    #         dataset=dataset,
    #         num_replicas=1,
    #         rank=0,
    #         shuffle=True if stage=='train' else False,
    #         seed=int(self.rng.integers(0, 2**32-1)),
    #         drop_last=True if stage=='train' else False,
    #     )
    #     batch_size: int = None
    #     if isinstance(self.batch_size, int):
    #         batch_size = self.batch_size
    #         pin_memory = True
    #     elif isinstance(self.batch_size, dict):
    #         pin_memory = False
    #         for key, value in reversed(self.batch_size.items()):
    #             if self.trainer.current_epoch >= key:
    #                 batch_size = value
    #                 break
    #     assert batch_size > 0
    #     num_workers = 0 if stage=='predict' else self.num_workers
    #     return torch.utils.data.DataLoader(
    #         dataset=dataset,
    #         sampler=sampler,
    #         batch_size=batch_size//self.trainer.world_size,  # batch size per rank
    #         num_workers=self.num_workers,
    #         drop_last=True if stage=='train' else False,
    #         prefetch_factor=2 if self.num_workers else None,
    #         pin_memory=pin_memory,
    #         persistent_workers=True if self.num_workers else False,
    #     )

    def set_fir_filter(self):
        assert len(self.fir_bp) == 2, "fir_bp must be a sequence of (f_low, f_high)"
        if self.fir_bp[0] or self.fir_bp[1]:
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
            loaders['elm_class'] = self.get_dataloader_from_dataset('train', self.elm_datasets['train'])
        if 'conf_onehot' in self.tasks:
            loaders['conf_onehot'] = self.get_dataloader_from_dataset('train', self.confinement_datasets['train'])
        combined_loader: CombinedLoader = CombinedLoader(
            iterables=loaders,
            mode='max_size_cycle',
        )
        _ = iter(combined_loader)
        return combined_loader

    def val_dataloader(self) -> CombinedLoader:
        loaders = {}
        if 'elm_class' in self.tasks:
            loaders['elm_class'] = self.get_dataloader_from_dataset('validation', self.elm_datasets['validation'])
        if 'conf_onehot' in self.tasks:
            loaders['conf_onehot'] = self.get_dataloader_from_dataset('validation', self.confinement_datasets['validation'])
        combined_loader: CombinedLoader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def test_dataloader(self) -> CombinedLoader:
        loaders = {}
        if 'elm_class' in self.tasks:
            loaders['elm_class'] = self.get_dataloader_from_dataset('test', self.elm_datasets['test'])
        if 'conf_onehot' in self.tasks:
            loaders['conf_onehot'] = self.get_dataloader_from_dataset('test', self.confinement_datasets['test'])
        combined_loader: CombinedLoader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def predict_dataloader(self) -> list:
        loaders = []
        self.loader_tasks = []
        for task in self.tasks:
            if task == 'elm_class':
                # self.rprint("  \u2B1C " + f" ELM n_datasets: {len(self.elm_datasets['predict'])}")
                for dataset in self.elm_datasets['predict']:
                    loaders.append(self.get_dataloader_from_dataset('predict', dataset))
                    self.loader_tasks.append(task)
            elif task == 'conf_onehot':
                # self.rprint("  \u2B1C " + f" conf. n_datasets: {len(self.confinement_datasets['predict'])}")
                for dataset in self.confinement_datasets['predict']:
                    loaders.append(self.get_dataloader_from_dataset('predict', dataset))
                    self.loader_tasks.append(task)
        self.zprint(f"  Predict dataloaders: {len(loaders)}")
        return loaders

    def load_state_dict(self, state: dict) -> None:
        self.zprint(f"  Loading state dict keys: {self.state_items}")
        for item in self.state_items:
            setattr(self, item, state[item])

    def save_state_dict(self):
        if self.is_global_zero:
            state_dict_file = Path(self.log_dir)/'state_dict.pt'
            state_dict_file.parent.mkdir(parents=True, exist_ok=True)
            self.zprint(f"Saving state_dict: {state_dict_file}")
            state_dict = {item: getattr(self, item) for item in self.state_items}
            for key in state_dict:
                self.zprint(f"    {key}")
            torch.save(state_dict, state_dict_file)

    def rprint(self, text: str = ''):
        self.barrier()
        super().rprint(text)
        self.barrier()

    def barrier(self):
        if self.world_size > 1:
            self.trainer.strategy.barrier()

    def broadcast(self, obj, src: int = 0):
        if self.world_size > 1:
            obj = self.trainer.strategy.broadcast(obj, src=src)
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
        signals = self.elms_to_signals_dict[sw_metadata['elm_index']]
        signal_window = signals[..., sw_metadata['i_t0'] : sw_metadata['i_t0'] + self.signal_window_size, :, :]
        quantile_binary_label = {q: int(sw_metadata['time_to_elm']<=qval) for q, qval in self.time_to_elm_quantiles.items()}
        return signal_window, quantile_binary_label, sw_metadata['elm_index'] #sw_metadata['time_to_elm']


@dataclasses.dataclass(eq=False)
class ELM_Predict_Dataset(torch.utils.data.Dataset):
    signal_window_size: int = 0
    signals: torch.Tensor = None # rank-wise signals (map to ELM indices)
    time: np.ndarray = None # rank-wise signals (map to ELM indices)
    interelm_times: tuple[float,float] = None # rank-wise signals (map to ELM indices)
    time_to_elm_quantiles: dict[float, float] = None
    shot: int = None
    elm_index: int = None
    stride_factor: int = 8

    def __post_init__(self):
        super().__init__()
        if self.time is None and self.signals is None:
            self.valid_t0 = []
            return
        self.t_start = self.interelm_times[0]
        self.t_stop = self.interelm_times[1]
        self.t_predict = self.time_to_elm_quantiles[0.5]
        self.i_start = np.flatnonzero(self.time >= self.t_start)[0]
        self.i_stop = np.flatnonzero(self.time < self.t_stop)[-1]
        self.i_predict = np.flatnonzero(self.time >= (self.t_stop-self.t_predict))[0]
        self.labels = np.zeros_like(self.time, dtype=int) * np.nan
        self.labels[self.i_start : self.i_predict] = 0
        self.labels[self.i_predict : self.i_stop+1] = 1
        i_sw_start = 0
        stride = self.signal_window_size // self.stride_factor
        self.valid_t0 = []
        while True:
            i_sw_end = i_sw_start + self.signal_window_size - 1
            if i_sw_end > self.time.size-1:
                break
            self.valid_t0.append(i_sw_start)
            i_sw_start += stride
        self.valid_t0 = np.array(self.valid_t0, dtype=int)
        assert self.valid_t0[-1] + self.signal_window_size - 1 <= self.time.size - 1

    def __len__(self) -> int:
        return len(self.valid_t0)
    
    def __getitem__(self, i: int) -> tuple:
        i_t0 = self.valid_t0[i]
        signal_window = self.signals[..., i_t0 : i_t0 + self.signal_window_size, :, :]
        time = self.time[i_t0 + self.signal_window_size - 1]
        label = self.labels[i_t0 + self.signal_window_size - 1]
        return signal_window, label, time

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
        # i_key = (self.window_start_indices <= i_t0).nonzero().max()
        # shot_event_key = self.shot_event_keys[i_key]
        # Convert the key to an integer by removing non-numeric characters and converting to int
        # confinement_mode_id_int = int(shot_event_key.replace('/', ''))
        # Convert to tensor
        # confinement_mode_id_tensor = torch.tensor([confinement_mode_id_int], dtype=torch.int64)
        return signal_window, label #confinement_mode_id_tensor


@dataclasses.dataclass(eq=False)
class Confinement_Predict_Dataset(torch.utils.data.Dataset):
    signals: np.ndarray = None
    labels: np.ndarray = None
    time: np.ndarray = None
    signal_window_size: int = None
    shot_event_keys: np.ndarray = None
    stride_factor: int = 8

    def __post_init__(self) -> None:
        if self.signals is None and self.time is None:
            self.sample_indices = torch.tensor([])
            return
        self.labels = torch.from_numpy(self.labels)
        self.signals = torch.from_numpy(self.signals[np.newaxis, ...])
        self.time = torch.from_numpy(self.time)
        self.shot = self.shot_event_keys[0]
        self.event = self.shot_event_keys[1]
        self.sample_indices = []
        start_index = 0
        while True:
            stop_index = start_index + self.signal_window_size - 1
            if stop_index > self.time.numel() - 1:
                break
            self.sample_indices.append(start_index)
            start_index += self.signal_window_size // self.stride_factor
        self.sample_indices = torch.tensor(self.sample_indices, dtype=torch.int64)

    def __len__(self) -> int:
        return self.sample_indices.numel()
    
    def __getitem__(self, i: int) -> tuple:
        # Retrieve the index from sample_indices that is guaranteed to have enough previous data
        i_start = self.sample_indices[i]
        # Define the start index for the signal window to look backwards
        i_stop = i_start + self.signal_window_size - 1
        # Retrieve the signal window from start_index to i_t0 (inclusive)
        signal_window = self.signals[..., i_start:i_stop+1, :, :]
        # The label is typically the current index in real-time scenarios
        label = self.labels[i_stop:i_stop+1]
        time = self.time[i_stop:i_stop+1]
        # Look up the correct confinement_mode_key based on i_t0
        # i_key = (self.window_start_indices <= i_start).nonzero().max()
        # shot_event_key = self.shot_event_keys[i_key]
        # Convert the key to an integer by removing non-numeric characters and converting to int
        # confinement_mode_id_int = int(shot_event_key.replace('/', ''))
        # Convert to tensor
        # confinement_mode_id_tensor = torch.tensor([confinement_mode_id_int], dtype=torch.int64)
        return signal_window, label, time


def main(
        experiment_name: str = 'experiment_default',
        trial_name: str = None,
        trial_name_prefix: str = None,
        restart_trial_name: str = None,
        wandb_id: str = None,
        signal_window_size: int = 256,
        # model
        no_bias: bool = False,
        batch_norm: bool = True,
        dropout: float = 0.0,
        feature_model_layers: Sequence[dict[str, LightningModule]] = None,
        mlp_tasks: dict[str, Sequence] = None,
        unfreeze_uncertainty_epoch: int = -1,
        # optimizer
        use_optimizer: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        deepest_layer_lr_factor: float = 1.0,
        lr_warmup_epochs: int = 0,
        lr_scheduler_patience: int = 100,
        lr_scheduler_threshold: float = 0.,
        monitor_metric = None,
        # transfer learning with backbone model
        backbone_model_path: str|Path = None,
        backbone_first_n_layers: int = None,
        backbone_initial_lr: float = 1e-3,
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
        skip_test: bool = False,
        skip_predict: bool = False,
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
        fir_bp: Sequence[float] = (None, None),
        confinement_dataset_factor: float = None,
        max_confinement_event_length: int = None,
        seed: int = 42,
        balance_confinement_data_with_elm_data: bool = False,
        bad_elm_indices: Sequence[int] = (),
) -> dict:

    ### SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    world_rank = int(os.getenv("SLURM_PROCID", default=0))

    is_global_zero = (world_rank == 0)
    def zprint(text):
        if is_global_zero:
            print(text)

    zprint(f"World rank {world_rank} of size {world_size} on {num_nodes} node(s)")

    ### model
    zprint("\u2B1C Creating model")
    lit_model = Model(
        signal_window_size=signal_window_size,
        no_bias=no_bias,
        batch_norm=batch_norm,
        feature_model_layers=feature_model_layers,
        mlp_tasks=mlp_tasks,
        dropout=dropout,
        # optimizer
        use_optimizer=use_optimizer,
        lr=lr,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_threshold=lr_scheduler_threshold,
        deepest_layer_lr_factor=deepest_layer_lr_factor,
        lr_warmup_epochs=lr_warmup_epochs,
        weight_decay=weight_decay,
        monitor_metric=monitor_metric,
        unfreeze_uncertainty_epoch=unfreeze_uncertainty_epoch,
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
            unfreeze_backbone_at_epoch=backbone_unfreeze_at_epoch,
            lambda_func=lambda _: backbone_warmup_rate,
            backbone_initial_lr=backbone_initial_lr,
            initial_denom_lr=1,
            train_bn=False,
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
        trial_name = f"r{slurm_identifier}" if slurm_identifier else f"r{datetime_str}"
        if trial_name_prefix:
            trial_name = f"{trial_name_prefix}_{trial_name}"
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
        0 if isinstance(batch_size, (int,np.int64)) else
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
        num_sanity_val_steps=2,
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
        if ckpt_path:
            assert seed is not None
        # TODO: after instantiating datamodule, load state_dict of checkpoint into datamodule
        lit_datamodule = Data(
            signal_window_size=signal_window_size,
            log_dir=trainer.log_dir,
            elm_data_file=elm_data_file,
            confinement_data_file=confinement_data_file,
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
            confinement_dataset_factor=confinement_dataset_factor,
            max_confinement_event_length=max_confinement_event_length,
            seed=seed,
            balance_confinement_data_with_elm_data=balance_confinement_data_with_elm_data,
            bad_elm_indices=bad_elm_indices,
        )

    if not skip_train and not skip_data:
        zprint("\u2B1C Begin Trainer.fit()")
        trainer.fit(
            model=lit_model, 
            datamodule=lit_datamodule,
            ckpt_path=ckpt_path,
        )
    if fraction_test and not skip_test and not skip_data:
        trainer.test(
            model=lit_model, 
            datamodule=lit_datamodule,
        )
    if fraction_test and not (skip_predict or skip_test) and not skip_data:
        trainer.predict(
            model=lit_model, 
            datamodule=lit_datamodule,
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
    seed = int(np.random.default_rng().integers(0, 2**32-1))
    print(seed)
    feature_model_layers = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        # {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        # {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
    )
    mlp_tasks={
        'elm_class': [None,16,1],
        'conf_onehot': [None,16,4],
    }
    # with open('count_of_bad_elms.pkl', 'rb') as f:
    #     data_dict = pickle.load(f)
    #     bad_elm_indices = data_dict['bad_elms']
    main(
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        # elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        # elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        # elm_data_file=ml_data.small_data_100,
        feature_model_layers=feature_model_layers,
        mlp_tasks=mlp_tasks,
        max_elms=150,
        max_epochs=3,
        # bad_elm_indices=bad_elm_indices,
        lr=1e-2,
        lr_warmup_epochs=2,
        batch_size=512,
        fraction_validation=0.125,
        fraction_test=0.125,
        num_workers=2,
        max_confinement_event_length=int(25e3),
        confinement_dataset_factor=0.25,
        # use_wandb=True,
        monitor_metric='sum_loss/train',
        seed=seed,
        # balance_confinement_data_with_elm_data=True,
        # skip_train=True,
        skip_test=True,
        # backbone_model_path='experiment_default/r2025_10_27_18_56_43',
        # backbone_unfreeze_at_epoch=1,
        # backbone_first_n_layers=3,
        # backbone_initial_lr=1e-3,
        # backbone_warmup_rate=2,
    )
