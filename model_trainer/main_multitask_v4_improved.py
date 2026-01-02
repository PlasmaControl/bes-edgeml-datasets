"""Improved copy of main_multitask_v4.py.

Changes in this copy (kept intentionally surgical):
- Make multiclass metrics use the configured number of classes (instead of hard-coding 4).
- Avoid assuming CUDA is available when placing metrics / selecting devices.
- Initialize per-step loss tensors on the correct device (prevents CPU/GPU mismatch).
- In predict plotting for multiclass, use softmax probabilities (not sigmoid/expit).
- Use a more portable Tensor -> numpy conversion in predict_step.

Notes / further suggestions (not fully refactored here to keep diffs small):
- Consider replacing the name-based task inference ("elm_class" vs "conf_onehot") with an explicit
    task spec (like TaskSpec in model_trainer/multi_objective_lightning.py) to remove magic strings.
- Consider logging/monitoring validation metrics by default (e.g., sum_loss/val) rather than train.
"""

from pathlib import Path
import contextlib
import dataclasses
from datetime import datetime
from typing import Literal, Optional, OrderedDict, Sequence, cast
import os
import sys
import time
import re
import pickle

import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import sklearn.metrics
import h5py

import torch
import torch.nn
import torch.cuda
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torch.multiprocessing as mp

from lightning.pytorch import Trainer, LightningModule, LightningDataModule, seed_everything
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import \
    LearningRateMonitor, EarlyStopping, ModelCheckpoint, BackboneFinetuning
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from torchmetrics import Metric
from torchmetrics.classification import \
    BinaryF1Score, BinaryPrecision, BinaryRecall, \
    MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import ml_data

torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)


# --- Stage contract (single source of truth) ---
# Sub-stage keys used throughout the codebase (dataset splits, metric suffixes, etc.)
STAGE_KEYS: tuple[str, ...] = ('train', 'val', 'test', 'predict')
STAGE_KEYS_FIT: tuple[str, ...] = ('train', 'val')
STAGE_KEYS_TRAIN_VAL_TEST: tuple[str, ...] = ('train', 'val', 'test')

# Lightning/Trainer stage values used by DataModule.setup()
TRAINER_STAGE_KEYS: tuple[str, ...] = ('fit', 'test', 'predict')


def _expit_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _firwin_windowed_sinc(
    *,
    numtaps: int,
    cutoff: float | list[float],
    pass_zero: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'],
    fs: float,
) -> np.ndarray:
    """SciPy-free FIR design for this codebase (Hamming-windowed sinc)."""

    if numtaps % 2 != 1:
        raise ValueError('numtaps must be odd')
    if fs <= 0:
        raise ValueError('fs must be positive')

    m = numtaps - 1
    n = np.arange(numtaps, dtype=np.float64)
    center = m / 2.0
    window = np.hamming(numtaps).astype(np.float64)

    def _ideal_lowpass(fc_hz: float) -> np.ndarray:
        fc = float(fc_hz) / float(fs)  # cycles/sample
        return (2.0 * fc) * np.sinc(2.0 * fc * (n - center))

    def _delta() -> np.ndarray:
        d = np.zeros(numtaps, dtype=np.float64)
        d[int(center)] = 1.0
        return d

    if isinstance(cutoff, (list, tuple, np.ndarray)):
        c = [float(v) for v in cutoff]
    else:
        c = [float(cutoff)]

    if pass_zero == 'lowpass':
        if len(c) != 1:
            raise ValueError('lowpass expects a single cutoff')
        h = _ideal_lowpass(c[0])
    elif pass_zero == 'highpass':
        if len(c) != 1:
            raise ValueError('highpass expects a single cutoff')
        h = _delta() - _ideal_lowpass(c[0])
    elif pass_zero == 'bandpass':
        if len(c) != 2:
            raise ValueError('bandpass expects two cutoffs')
        f1, f2 = sorted(c)
        h = _ideal_lowpass(f2) - _ideal_lowpass(f1)
    elif pass_zero == 'bandstop':
        if len(c) != 2:
            raise ValueError('bandstop expects two cutoffs')
        f1, f2 = sorted(c)
        h = _delta() - (_ideal_lowpass(f2) - _ideal_lowpass(f1))
    else:
        raise ValueError(f'Unknown pass_zero={pass_zero!r}')

    h = h * window
    if pass_zero in ('lowpass', 'bandstop'):
        s = float(np.sum(h))
        if s != 0.0:
            h = h / s
    return h.astype(np.float32)


def _lfilter_fir_causal(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Causal FIR filtering y[n]=sum_k b[k] x[n-k] along the last axis."""

    x = np.asarray(x)
    b = np.asarray(b)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    def _conv_1d(sig_1d: np.ndarray) -> np.ndarray:
        y_full = np.convolve(sig_1d, b, mode='full')
        return y_full[: sig_1d.size]

    return np.apply_along_axis(_conv_1d, axis=-1, arr=x).astype(np.float32, copy=False)


def _validate_sub_stage_key(stage: str) -> str:
    if stage not in STAGE_KEYS:
        raise ValueError(f"Invalid sub-stage key: {stage!r}. Expected one of {STAGE_KEYS}")
    return stage


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


TaskType = Literal["binary", "multiclass"]
LabelFormat = Literal["binary_quantile_dict", "binary_logit", "multiclass_index"]


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Explicit task specification.

    This keeps task behavior out of magic-name conditionals.

    Notes
    - For this script's current datasets:
      - ELM classification labels come as a dict keyed by quantile (e.g. 0.5).
      - Confinement labels are integer class indices.
    """

    name: str
    task_type: TaskType
    head_layers: Sequence[int]
    # Tasks sharing the same input stream can share the trunk forward pass.
    # If None, defaults to `name` (no sharing).
    input_key: Optional[str] = None
    num_classes: Optional[int] = None
    label_format: LabelFormat = "binary_logit"
    label_quantile: Optional[float] = None
    dataloader_idx: Optional[int] = None
    monitor_metric: str = "f1_score/val"
    class_labels: Optional[Sequence[str]] = None
    track_elmwise_f1: bool = False


def _validate_task_specs(task_specs: Sequence[TaskSpec]) -> tuple[TaskSpec, ...]:
    if task_specs is None:
        task_specs = _default_task_specs()

    if not task_specs:
        raise ValueError("task_specs must be a non-empty sequence")

    if not all([hasattr(spec, 'name') for spec in task_specs]):
        raise ValueError("Each TaskSpec must have a 'name' attribute")

    names = [spec.name for spec in task_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate task names in task_specs: {names}")

    for spec in task_specs:
        if spec.input_key is not None and not str(spec.input_key).strip():
            raise ValueError(f"TaskSpec.input_key must be non-empty when provided (task={spec.name!r})")

    dataloader_idxs = [spec.dataloader_idx for spec in task_specs]
    has_any_dataloader_idx = any(idx is not None for idx in dataloader_idxs)
    if has_any_dataloader_idx:
        if not all(idx is not None for idx in dataloader_idxs):
            raise ValueError("If any TaskSpec.dataloader_idx is set, all must be set")

        dataloader_idxs_int = [int(idx) for idx in dataloader_idxs]
        if len(set(dataloader_idxs_int)) != len(dataloader_idxs_int):
            raise ValueError(f"Duplicate dataloader_idx in task_specs: {dataloader_idxs_int}")

        expected = list(range(len(dataloader_idxs_int)))
        if sorted(dataloader_idxs_int) != expected:
            raise ValueError(
                "TaskSpec.dataloader_idx must be a contiguous 0..N-1 set; "
                f"got {dataloader_idxs_int}"
            )

        if dataloader_idxs_int != expected:
            raise ValueError(
                "task_specs order must match dataloader_idx order; "
                f"expected {expected} but got {dataloader_idxs_int}"
            )

    return tuple(task_specs)


def _default_task_specs() -> tuple[TaskSpec, ...]:
    """Default task configuration (single-task ELM classifier)."""

    return (
        TaskSpec(
            name='elm_class',
            task_type='binary',
            head_layers=(16,),
            label_format='binary_quantile_dict',
            label_quantile=0.5,
            dataloader_idx=0,
            track_elmwise_f1=True,
        ),
    )


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
    # batch_norm: bool = True
    dropout: float = 0.05
    feature_model_layers: Sequence[dict[str, LightningModule]] = None
    task_specs: Sequence[TaskSpec] = None
    unfreeze_logsigma_epoch: int = -1
    logsigma_warmup_epochs: int = 0
    multiobjective_method: Literal['logsigma', 'pcgrad', 'gradnorm'] = 'logsigma'
    gradnorm_alpha: float = 1.5
    gradnorm_eta: float = 0.025
    # latent_batch_norm: bool = True
    grad_update_interval: int = 1
    gradnorm_rep_params: Literal['all', 'last_trunk_layer'] = 'last_trunk_layer'
    use_torch_compile: bool = False
    backbone_model_path: str|Path = None
    backbone_first_n_layers: int = None
    elmwise_f1_interval: int = 25

    def __post_init__(self):

        # init superclasses
        _Base_Class.__post_init__(self)
        LightningModule.__init__(self)
        # Avoid logging task_specs here; we finalize and log it later in a serializable form.
        self.save_hyperparameters(ignore=['task_specs'])
        self.trainer: Trainer = None
        self.run_dir: Path = None
        # Avoid assuming CUDA exists; Lightning will move modules/metrics to the correct device.
        self.local_device: str = (
            f'cuda:{self.slurm_local_rank}' if torch.cuda.is_available() else 'cpu'
        )

        if self.is_global_zero:
            print_fields(self)

        # input data shape
        self.input_data_shape = (1, 1, self.signal_window_size, 8, 8)

        # feature space sub-model
        self.feature_model: LightningModule = None
        self.feature_space_size: int = None
        self.make_feature_model()
        assert self.feature_space_size

        # --- Task configuration ---
        # TaskSpec is the only supported external task configuration.
        self.task_specs: Sequence[TaskSpec] = _validate_task_specs(self.task_specs)

        # Log a JSON-friendly representation of task_specs (and only from the model) to avoid
        # Lightning's hparams merge conflict with the DataModule.
        self.save_hyperparameters({'task_specs': [dataclasses.asdict(spec) for spec in self.task_specs]})

        self.task_specs_by_name: dict[str, TaskSpec] = {spec.name: spec for spec in self.task_specs}
        # (duplicate names are already checked in _validate_task_specs)

        # Canonical ordered list of task names.
        self.task_names: tuple[str, ...] = tuple(spec.name for spec in self.task_specs)

        self.is_multitask = len(self.task_specs) > 1

        # Task grouping for shared trunk computation.
        self.task_input_key: dict[str, str] = {
            spec.name: (str(spec.input_key) if spec.input_key is not None else spec.name)
            for spec in self.task_specs
        }
        self.input_key_to_tasks: dict[str, list[str]] = {}
        for task_name, key in self.task_input_key.items():
            self.input_key_to_tasks.setdefault(key, []).append(task_name)

        if self.logsigma_warmup_epochs and self.unfreeze_logsigma_epoch == -1:
            self.unfreeze_logsigma_epoch = int(self.logsigma_warmup_epochs)

        # create MLP configs + metrics
        self.elm_wise_results = {}
        self.task_configs = {}
        for spec in self.task_specs:
            self.task_configs[spec.name] = {}
            if spec.task_type == 'binary':
                out_dim = 1
            elif spec.task_type == 'multiclass':
                if spec.num_classes is None or int(spec.num_classes) <= 1:
                    raise ValueError(f"Task '{spec.name}' multiclass requires num_classes > 1")
                out_dim = int(spec.num_classes)
            else:
                raise ValueError(f"Unsupported task_type: {spec.task_type}")

            self.task_configs[spec.name]['layers'] = [
                int(self.feature_space_size),
                *[int(x) for x in spec.head_layers],
                int(out_dim),
            ]
            if spec.task_type == 'binary':
                self.task_configs[spec.name]['score_metric_names'] = (
                    'f1_score',
                    'precision_score',
                    'recall_score',
                )
                self.task_configs[spec.name]['metrics'] = {
                    'bce_loss': torch.nn.functional.binary_cross_entropy_with_logits,
                    'mean_stat': torch.mean,
                    'std_stat': torch.std,
                }
            elif spec.task_type == 'multiclass':
                num_classes = int(spec.num_classes)
                self.task_configs[spec.name]['score_metric_names'] = (
                    # Back-compat defaults (treat as macro):
                    'f1_score',
                    'precision_score',
                    'recall_score',
                    # Explicit macro/micro variants:
                    'f1_macro',
                    'f1_micro',
                    'precision_macro',
                    'precision_micro',
                    'recall_macro',
                    'recall_micro',
                )
                self.task_configs[spec.name]['metrics'] = {
                    'ce_loss': torch.nn.functional.cross_entropy,
                    'mean_stat': lambda t: torch.abs(torch.mean(t)),
                    'std_stat': torch.std,
                }
            self.task_configs[spec.name]['monitor_metric'] = spec.monitor_metric

        # make task sub-models
        self.task_metrics: dict[str, dict] = {}
        self.task_models: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self.task_log_sigma: torch.nn.ParameterDict = torch.nn.ParameterDict()
        for task_name, task_dict in self.task_configs.items():
            # task_dict = self.task_configs[task_name]
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

        # TorchMetrics must be registered as modules to be moved with the model and to keep
        # separate state per stage (train/val/test). Do not reuse the same Metric object across
        # different stages.
        self.stage_task_metrics: torch.nn.ModuleDict = torch.nn.ModuleDict()
        for spec in self.task_specs:
            task_name = spec.name
            score_metric_names: tuple[str, ...] = tuple(self.task_configs[task_name].get('score_metric_names', ()))
            for stage_key in STAGE_KEYS_TRAIN_VAL_TEST:
                for metric_name in score_metric_names:
                    key = f"{task_name}__{stage_key}__{metric_name}"
                    if spec.task_type == 'binary':
                        metric: torch.nn.Module
                        if metric_name == 'f1_score':
                            metric = BinaryF1Score()
                        elif metric_name == 'precision_score':
                            metric = BinaryPrecision()
                        elif metric_name == 'recall_score':
                            metric = BinaryRecall()
                        else:
                            raise ValueError(f"Unknown binary score metric: {metric_name}")
                    elif spec.task_type == 'multiclass':
                        if spec.num_classes is None or int(spec.num_classes) <= 1:
                            raise ValueError(f"Task '{task_name}' multiclass requires num_classes > 1")
                        num_classes = int(spec.num_classes)
                        average = 'micro' if metric_name.endswith('_micro') else 'macro'
                        if metric_name in ('f1_score', 'f1_macro', 'f1_micro'):
                            metric = MulticlassF1Score(num_classes=num_classes, average=average)
                        elif metric_name in ('precision_score', 'precision_macro', 'precision_micro'):
                            metric = MulticlassPrecision(num_classes=num_classes, average=average)
                        elif metric_name in ('recall_score', 'recall_macro', 'recall_micro'):
                            metric = MulticlassRecall(num_classes=num_classes, average=average)
                        else:
                            raise ValueError(f"Unknown multiclass score metric: {metric_name}")
                    else:
                        raise ValueError(f"Unsupported task_type: {spec.task_type}")

                    self.stage_task_metrics[key] = metric

        # Optional BatchNorm1d on the latent feature vector before task heads.
        # self.latent_bn: Optional[torch.nn.BatchNorm1d] = (
        #     torch.nn.BatchNorm1d(int(self.feature_space_size)) if self.latent_batch_norm else None
        # )

        # Cache shared parameters (trunk + optional latent BN) and refresh only when trainability changes.
        self._shared_params_cache_signature: Optional[tuple[tuple[int, bool], ...]] = None
        self._cached_shared_params: list[torch.nn.Parameter] = []
        self._cached_trunk_params: list[torch.nn.Parameter] = []
        self._cached_gradnorm_rep_params: list[torch.nn.Parameter] = []
        self._cached_gradnorm_rep_signature: Optional[tuple[tuple[int, bool], ...]] = None
        self._refresh_shared_params_cache(force=True)

        self._torch_compile_done: bool = False

        # Multi-objective training mode.
        if self.multiobjective_method not in ('logsigma', 'pcgrad', 'gradnorm'):
            raise ValueError(f"Unknown multiobjective_method: {self.multiobjective_method}")
        # PCGrad/GradNorm require manual optimization.
        self.automatic_optimization = (self.multiobjective_method == 'logsigma')

        # GradNorm-style task weights stored as buffers for checkpointing.
        # Initialized lazily (only when used) to keep default behavior unchanged.
        if self.multiobjective_method == 'gradnorm' and self.is_multitask:
            for task_name in self.task_names:
                w = torch.tensor(1.0)
                w.requires_grad_(True)
                self.register_buffer(f"gradnorm_w__{task_name}", w, persistent=True)
                self.register_buffer(f"gradnorm_L0__{task_name}", torch.tensor(float('nan')), persistent=True)

        self.task_log_sigma.requires_grad_ = True if self.unfreeze_logsigma_epoch == -1 else False

        self.zprint(f"Total model parameters: {self.param_count(self):,d}")

        self.initialize_parameters()
        if self.backbone_model_path and self.backbone_first_n_layers:
            self.backbone_transfer_learning()

    def make_feature_model(self) -> None:
        self.zprint("Feature space sub-model")
        if self.feature_model_layers is None:
            self.feature_model_layers = (
                {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
                {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1},
                {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1)},
            )
        n_feature_layers = len(self.feature_model_layers)
        feature_layer_dict = OrderedDict()
        data_shape = self.input_data_shape
        self.zprint(f"  Input data shape: {data_shape}  (size {np.prod(data_shape)})")
        previous_out_channels: int = None
        for i_layer, layer in enumerate(self.feature_model_layers):
            in_channels: int = 1 if i_layer==0 else previous_out_channels

            # conv layer
            conv = torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=layer['out_channels'],
                kernel_size=layer['kernel'],
                stride=layer['stride'],
                bias=False,
            )
            conv_layer_name = f"L{i_layer:02d}_Conv"
            feature_layer_dict[conv_layer_name] = conv
            n_params = self.param_count(conv)
            data_shape = tuple(conv(torch.zeros(data_shape)).shape)
            data_size = np.prod(data_shape)
            self.zprint(f"  {conv_layer_name} kern {conv.kernel_size}  stride {conv.stride}  out_ch {conv.out_channels}  param {n_params:,d}  output {data_shape} (size {data_size})")
            previous_out_channels = conv.out_channels

            # batchnorm after conv
            # layer_name = f"L{i_layer:02d}_BatchNorm"
            # feature_layer_dict[layer_name] = torch.nn.BatchNorm3d(in_channels)
            layer_name = f"L{i_layer:02d}_GroupNorm"
            feature_layer_dict[layer_name] = torch.nn.GroupNorm(1, conv.out_channels)
            self.zprint(f"  {layer_name} (regularization)")

            # LeakyReLU after batchnorm
            relu_layer_name = f"L{i_layer:02d}_LeakyReLU"
            feature_layer_dict[relu_layer_name] = torch.nn.LeakyReLU(self.leaky_relu_slope)
            self.zprint(f"  {relu_layer_name} (activation)")

        # flatten and groupnorm
        layer_name = 'Feature_Space_Flatten'
        feature_layer_dict[layer_name] = torch.nn.Flatten()
        self.zprint(f"  {layer_name}  (flatten to vector)")
        layer_name = "Feature_Space_LayerNorm"
        feature_layer_dict[layer_name] = torch.nn.LayerNorm(data_size)
        self.zprint(f"  {layer_name}  (regularization)")

        self.feature_model = torch.nn.Sequential(feature_layer_dict)
        self.feature_space_size = self.feature_model(torch.zeros(self.input_data_shape)).numel()
        assert self.feature_space_size == data_size

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
            # fully-connected layer
            # bias = False if i_mlp_layer<n_mlp_layers-2 else True
            # Linear
            mlp_layer = torch.nn.Linear(
                in_features=mlp_layers[i_mlp_layer],
                out_features=mlp_layers[i_mlp_layer+1],
                bias=False if i_mlp_layer<n_mlp_layers-2 else True,
            )
            layer_name = f"L{i_layer:02d}_FC"
            mlp_layer_dict[layer_name] = mlp_layer
            n_params = self.param_count(mlp_layer)
            self.zprint(f"  {layer_name}  in_features {mlp_layer.in_features}  out_features {mlp_layer.out_features}  parameters {n_params:,d}")

            if i_mlp_layer < n_mlp_layers-2:
                # batchnorm after Linear
                layer_name = f"L{i_layer:02d}_BatchNorm"
                mlp_layer_dict[layer_name] = torch.nn.BatchNorm1d(mlp_layers[i_mlp_layer+1])
                self.zprint(f'  {layer_name} (regularization)')

                # leaky relu
                layer_name = f"L{i_layer:02d}_LeakyReLU"
                mlp_layer_dict[layer_name] = torch.nn.LeakyReLU(self.leaky_relu_slope)
                self.zprint(f"  {layer_name} (activation)")

                # dropout
                layer_name = f'L_{i_layer:02d}_Dropout'
                mlp_layer_dict[layer_name] = torch.nn.Dropout1d(p=self.dropout,inplace=True)
                self.zprint(f"  {layer_name} (regularization)")

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
                spec.name: [torch.randn(
                    size=[512]+list(self.input_data_shape[1:]),
                    dtype=torch.float32,
                )]
                for spec in self.task_specs
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
        self.zprint(f"  Unfreeze logsigma epoch: {self.unfreeze_logsigma_epoch:d}")

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
                # Always include sigma params in the optimizer so we can unfreeze them later
                # without modifying optimizer param groups.
                params_sigmas['params'].append(p)
                if self.unfreeze_logsigma_epoch == -1:
                    p.requires_grad_(True)
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
        # In manual optimization, Lightning will not auto-step schedulers.
        # Avoid Lightning's warning about ignored keys by omitting 'monitor' and step manually.
        schedulers = [
            (
                {'scheduler': plateau_scheduler, 'monitor': self.monitor_metric}
                if self.automatic_optimization
                else {'scheduler': plateau_scheduler}
            ),
        ]
        if self.lr_warmup_epochs:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=0.05,
                total_iters=self.lr_warmup_epochs,
            )
            schedulers.append({'scheduler': warmup_scheduler})
        return ( [optimizer], schedulers, )

    def _get_logged_metric_value(self, metric_name: str) -> Optional[float]:
        if self.trainer is None:
            return None
        logged = getattr(self.trainer, 'logged_metrics', None)
        if not isinstance(logged, dict) or metric_name not in logged:
            return None
        v = logged[metric_name]
        if isinstance(v, torch.Tensor):
            if v.numel() == 0:
                return None
            v = v.detach().float().mean().cpu().item()
        try:
            return float(v)
        except Exception:
            return None

    def _manual_step_lr_schedulers(self, where: Literal['train_epoch_end', 'val_epoch_end']) -> None:
        """Step LR schedulers in manual optimization modes.

        - ReduceLROnPlateau is stepped with the configured monitor metric.
        - Other schedulers (e.g., warmup LinearLR) are stepped without args.
        """

        if self.automatic_optimization:
            return
        if self.trainer is None:
            return

        scheds = self.lr_schedulers()
        if scheds is None:
            return
        if not isinstance(scheds, list):
            scheds = [scheds]

        monitor = self.monitor_metric
        monitor_value = self._get_logged_metric_value(monitor) if monitor else None

        # Decide which hook should step the plateau scheduler.
        prefer_val = bool(monitor and '/val' in monitor)
        prefer_train = bool(monitor and '/train' in monitor)
        should_step_plateau_here = (
            (prefer_val and where == 'val_epoch_end')
            or (prefer_train and where == 'train_epoch_end')
            or ((not prefer_val and not prefer_train) and where == 'val_epoch_end')
        )

        # Prevent double-stepping within the same epoch.
        last_epoch = getattr(self, '_last_manual_scheduler_epoch', None)
        last_where = getattr(self, '_last_manual_scheduler_where', None)
        if last_epoch == self.current_epoch and last_where == where:
            return

        def _max_lr() -> Optional[float]:
            if self.trainer is None:
                return None
            opts = getattr(self.trainer, 'optimizers', None)
            if not opts:
                return None
            lrs: list[float] = []
            for opt in opts:
                for pg in getattr(opt, 'param_groups', []) or []:
                    lr = pg.get('lr', None)
                    if lr is not None:
                        try:
                            lrs.append(float(lr))
                        except Exception:
                            pass
            return max(lrs) if lrs else None

        def _fmt_lr(v: Optional[float]) -> str:
            if v is None:
                return 'None'
            return f"{v:.2e}"

        def _lr_changed(before: Optional[float], after: Optional[float]) -> bool:
            if before is None or after is None:
                return False
            # Use a tiny relative tolerance to avoid floating noise.
            denom = max(abs(before), abs(after), 1e-12)
            return (abs(after - before) / denom) > 1e-9

        for s in scheds:
            if isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not should_step_plateau_here:
                    continue
                if monitor_value is None:
                    continue
                lr_before = _max_lr()
                s.step(monitor_value)
                lr_after = _max_lr()
                if self.is_global_zero and _lr_changed(lr_before, lr_after):
                    self.zprint(
                        f"LR step ({where}) ReduceLROnPlateau "
                        f"monitor={monitor}={monitor_value:.4g} "
                        f"lr { _fmt_lr(lr_before) } -> { _fmt_lr(lr_after) }"
                    )
            else:
                # Warmup schedulers are expected to be epoch-stepped.
                if where != 'train_epoch_end':
                    continue
                if self.lr_warmup_epochs and self.current_epoch >= int(self.lr_warmup_epochs):
                    continue
                lr_before = _max_lr()
                s.step()
                lr_after = _max_lr()
                if self.is_global_zero and _lr_changed(lr_before, lr_after):
                    self.zprint(
                        f"LR step ({where}) {s.__class__.__name__} "
                        f"lr { _fmt_lr(lr_before) } -> { _fmt_lr(lr_after) }"
                    )

        self._last_manual_scheduler_epoch = int(self.current_epoch)
        self._last_manual_scheduler_where = where

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

    def training_step(self, batch, batch_idx, dataloader_idx=None) -> torch.Tensor | Sequence[torch.Tensor]:
        if not (self.current_epoch or batch_idx or dataloader_idx):
            self.zprint("Begin training")

        # Unfreeze logsigma parameters after warm start.
        if (
            self.multiobjective_method == 'logsigma'
            and self.is_multitask
            and self.unfreeze_logsigma_epoch != -1
            and self.current_epoch == self.unfreeze_logsigma_epoch
        ):
            for p in self.task_log_sigma.parameters():
                p.requires_grad_(True)

        if self.automatic_optimization:
            output = self.update_step(
                batch,
                batch_idx,
                stage='train',
                dataloader_idx=dataloader_idx,
            )
            return output

        # Manual optimization path for PCGrad/GradNorm.
        optimizers = self.optimizers()
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        optimizer.zero_grad(set_to_none=True)

        sum_loss, task_losses = self.update_step(
            batch,
            batch_idx,
            stage='train',
            dataloader_idx=dataloader_idx,
            return_task_losses=True,
        )

        shared_params = self._get_shared_params()

        if self.multiobjective_method == 'pcgrad':
            self._pcgrad_step(task_losses=task_losses, shared_params=shared_params)
        elif self.multiobjective_method == 'gradnorm':
            self._gradnorm_step(task_losses=task_losses, shared_params=shared_params)
        else:
            raise ValueError(f"Unknown multiobjective_method: {self.multiobjective_method}")

        optimizer.step()
        return sum_loss

    def _ddp_manual_grad_sync_is_possible(self) -> bool:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return False
        if self.trainer is None:
            return False
        strategy = getattr(self.trainer, "strategy", None)
        return bool(strategy is not None and hasattr(strategy, "block_backward_sync"))

    def _ddp_allreduce_grads_once(self, params: Sequence[torch.nn.Parameter]) -> None:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        ws = torch.distributed.get_world_size()
        seen: set[int] = set()
        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if p.grad is None:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(ws)

    def _shared_params_signature(self) -> tuple[tuple[int, bool], ...]:
        items: list[tuple[int, bool]] = []
        for p in self.feature_model.parameters():
            items.append((id(p), bool(p.requires_grad)))
        # if self.latent_bn is not None:
        #     for p in self.latent_bn.parameters():
        #         items.append((id(p), bool(p.requires_grad)))
        return tuple(items)

    def _refresh_shared_params_cache(self, *, force: bool = False) -> None:
        sig = self._shared_params_signature()
        if (not force) and self._shared_params_cache_signature == sig:
            return
        self._shared_params_cache_signature = sig
        self._cached_trunk_params = [p for p in self.feature_model.parameters() if p.requires_grad]
        shared = list(self._cached_trunk_params)
        # if self.latent_bn is not None:
        #     shared.extend([p for p in self.latent_bn.parameters() if p.requires_grad])
        self._cached_shared_params = shared
        # Invalidate GradNorm rep cache.
        self._cached_gradnorm_rep_signature = None
        self._cached_gradnorm_rep_params = []

    def _get_shared_params(self) -> list[torch.nn.Parameter]:
        self._refresh_shared_params_cache(force=False)
        return self._cached_shared_params

    def _get_gradnorm_rep_params(self) -> list[torch.nn.Parameter]:
        self._refresh_shared_params_cache(force=False)
        if self.gradnorm_rep_params == 'all':
            return self._cached_shared_params

        sig = self._shared_params_cache_signature
        if sig is not None and self._cached_gradnorm_rep_signature == sig and self._cached_gradnorm_rep_params:
            return self._cached_gradnorm_rep_params

        rep: list[torch.nn.Parameter] = []
        last_with_params: Optional[torch.nn.Module] = None
        for _, child in reversed(list(self.feature_model.named_children())):
            if any(p.requires_grad for p in child.parameters(recurse=True)):
                last_with_params = child
                break
        if last_with_params is not None:
            rep = [p for p in last_with_params.parameters(recurse=True) if p.requires_grad]
        if not rep:
            rep = list(self._cached_trunk_params)

        self._cached_gradnorm_rep_params = rep
        self._cached_gradnorm_rep_signature = sig
        return rep

    def _task_shuffle_generator(self) -> torch.Generator:
        """Deterministic per-step RNG shared across ranks."""

        g = torch.Generator(device='cpu')
        base_seed = 0
        dm = getattr(getattr(self, 'trainer', None), 'datamodule', None)
        if dm is not None and getattr(dm, 'seed', None) is not None:
            base_seed = int(dm.seed)
        step = int(getattr(self, 'global_step', 0))
        g.manual_seed((base_seed + step) % (2**63 - 1))
        return g

    def _pcgrad_step(self, task_losses: dict[str, torch.Tensor], shared_params: list[torch.nn.Parameter]) -> None:
        tasks = [t for t in self.task_names if t in task_losses]
        if len(tasks) <= 1 or not shared_params:
            total_loss = sum(task_losses.values())
            self.manual_backward(total_loss)
            return

        interval = int(self.grad_update_interval) if self.grad_update_interval else 1
        interval = max(1, interval)
        if interval > 1 and (int(getattr(self, 'global_step', 0)) % interval) != 0:
            # Cheap step: skip PCGrad projection.
            total_loss = sum(task_losses.values())
            do_manual_ddp_sync = self._ddp_manual_grad_sync_is_possible()
            backward_sync_context = (
                self.trainer.strategy.block_backward_sync() if do_manual_ddp_sync else contextlib.nullcontext()
            )
            with backward_sync_context:
                self.manual_backward(total_loss)
            if do_manual_ddp_sync:
                self._ddp_allreduce_grads_once([p for p in self.parameters() if p.requires_grad])
            return

        # Stochastic (but deterministic) projection order per step.
        gen = self._task_shuffle_generator()
        perm = torch.randperm(len(tasks), generator=gen).tolist()
        tasks = [tasks[i] for i in perm]

        # Per-task gradients on shared parameters (flattened once per task for fast dot products).
        sizes = [int(p.numel()) for p in shared_params]
        dtypes = [p.dtype for p in shared_params]
        grads_by_task_flat: dict[str, torch.Tensor] = {}
        for i, t in enumerate(tasks):
            retain = i != (len(tasks) - 1)
            grads = torch.autograd.grad(
                task_losses[t],
                shared_params,
                retain_graph=retain,
                allow_unused=True,
            )
            flat_parts: list[torch.Tensor] = []
            for j, (g, n) in enumerate(zip(grads, sizes)):
                if g is None:
                    flat_parts.append(torch.zeros((n,), device=self.device, dtype=dtypes[j]))
                else:
                    flat_parts.append(g.reshape(-1))
            grads_by_task_flat[t] = torch.cat(flat_parts, dim=0)

        # Project conflicting gradients (vector math).
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

        # Unflatten merged grads back into parameter-shaped tensors.
        merged_grads: list[torch.Tensor] = []
        offset = 0
        for p, n in zip(shared_params, sizes):
            merged_grads.append(merged_flat[offset: offset + n].reshape_as(p))
            offset += n

        # Backprop full loss to get head gradients.
        # In DDP, skip the implicit sync here and do exactly one explicit all-reduce
        # after we overwrite the shared grads.
        total_loss = sum(task_losses.values())
        do_manual_ddp_sync = self._ddp_manual_grad_sync_is_possible()
        backward_sync_context = (
            self.trainer.strategy.block_backward_sync() if do_manual_ddp_sync else contextlib.nullcontext()
        )
        with backward_sync_context:
            self.manual_backward(total_loss)

        # Override shared grads with PCGrad merged grads.
        for p, g in zip(shared_params, merged_grads):
            p.grad = g.detach()

        # In DDP, synchronize all grads exactly once (shared + heads).
        if do_manual_ddp_sync:
            self._ddp_allreduce_grads_once([p for p in self.parameters() if p.requires_grad])

    def _gradnorm_step(self, task_losses: dict[str, torch.Tensor], shared_params: list[torch.nn.Parameter]) -> None:
        tasks = [t for t in self.task_names if t in task_losses]
        if len(tasks) <= 1 or not shared_params:
            total_loss = sum(task_losses.values())
            self.manual_backward(total_loss)
            return

        interval = int(self.grad_update_interval) if self.grad_update_interval else 1
        interval = max(1, interval)
        do_update = (interval <= 1) or ((int(getattr(self, 'global_step', 0)) % interval) == 0)

        eps = 1e-8

        # Initialize L0 on the first step (use reduced losses so all ranks match).
        detached_losses = {t: task_losses[t].detach() for t in tasks}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
            for t in tasks:
                tmp = detached_losses[t].clone()
                torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.SUM)
                detached_losses[t] = tmp / ws

        for t in tasks:
            L0_buf = getattr(self, f"gradnorm_L0__{t}")
            if torch.isnan(L0_buf):
                L0_buf.copy_(detached_losses[t].clamp_min(eps))

        if do_update:
            rep_params = self._get_gradnorm_rep_params()

            # Compute gradient norms per task (keep graph for gradients w.r.t. w).
            g_norms: dict[str, torch.Tensor] = {}
            w_vars: dict[str, torch.Tensor] = {}
            for t in tasks:
                w_raw = getattr(self, f"gradnorm_w__{t}")
                w = w_raw.clamp_min(1e-3)
                w_vars[t] = w_raw
                grads = torch.autograd.grad(
                    w * task_losses[t],
                    rep_params,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )
                sq = torch.zeros((), device=self.device)
                for g in grads:
                    if g is None:
                        continue
                    sq = sq + (g * g).sum()
                g_norms[t] = torch.sqrt(sq + eps)

            # Build target G* using globally consistent (detached) stats.
            g_norms_detached = {t: g_norms[t].detach() for t in tasks}
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws = torch.distributed.get_world_size()
                for t in tasks:
                    tmp = g_norms_detached[t].clone()
                    torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.SUM)
                    g_norms_detached[t] = tmp / ws

            g_avg = torch.mean(torch.stack([g_norms_detached[t] for t in tasks]))
            r = torch.stack([
                (detached_losses[t] / getattr(self, f"gradnorm_L0__{t}").clamp_min(eps)) ** float(self.gradnorm_alpha)
                for t in tasks
            ])
            r = r / (r.mean().clamp_min(eps))
            target = (g_avg * r).detach()

            # Optimize w by gradient descent on sum |G_i - G*_i|.
            gradnorm_obj = torch.zeros((), device=self.device)
            for i, t in enumerate(tasks):
                gradnorm_obj = gradnorm_obj + torch.abs(g_norms[t] - target[i])

            w_list = [w_vars[t] for t in tasks]
            grads_w = torch.autograd.grad(
                gradnorm_obj,
                w_list,
                retain_graph=True,
                allow_unused=True,
            )

            # Average w-grads across ranks so every rank applies the same update.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws = torch.distributed.get_world_size()
                grads_w = list(grads_w)
                for i, g in enumerate(grads_w):
                    if g is None:
                        continue
                    tmp = g.detach().clone()
                    torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.SUM)
                    grads_w[i] = tmp / ws

            with torch.no_grad():
                lr_w = float(self.gradnorm_eta)
                for t, g in zip(tasks, grads_w):
                    if g is None:
                        continue
                    w_buf = getattr(self, f"gradnorm_w__{t}")
                    w_buf.add_(-lr_w * g)
                    w_buf.clamp_(min=1e-3)

                # Renormalize weights to sum to number of tasks.
                w_sum = torch.sum(torch.stack([getattr(self, f"gradnorm_w__{t}") for t in tasks]))
                scale = (len(tasks) / w_sum.clamp_min(eps))
                for t in tasks:
                    getattr(self, f"gradnorm_w__{t}").mul_(scale)

        total_loss = sum(getattr(self, f"gradnorm_w__{t}").detach() * task_losses[t] for t in tasks)
        do_manual_ddp_sync = self._ddp_manual_grad_sync_is_possible()
        backward_sync_context = (
            self.trainer.strategy.block_backward_sync() if do_manual_ddp_sync else contextlib.nullcontext()
        )
        with backward_sync_context:
            self.manual_backward(total_loss)

        if do_manual_ddp_sync:
            self._ddp_allreduce_grads_once([p for p in self.parameters() if p.requires_grad])

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
        # if self.latent_bn is not None:
        #     features = self.latent_bn(features)
        datamodule: Data = self.trainer.datamodule
        task = datamodule.loader_tasks[dataloader_idx]
        task_outputs = self.task_models[task](features)
        prediction_outputs = {
            'outputs': task_outputs.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy(),
            'times': times.detach().cpu().numpy(),
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
            return_task_losses: bool = False,
    ) -> torch.Tensor | Sequence:
        sync_dist_epoch = bool(getattr(getattr(self, 'trainer', None), 'world_size', 1) > 1)
        # Keep loss tensors on the same device as model outputs.
        sum_loss = torch.zeros((), device=self.device)
        prod_loss = torch.ones((), device=self.device)
        task_losses: dict[str, torch.Tensor] = {}
        model_outputs = self(batch)

        # CombinedLoader yields dict-like batches; Lightning can't always infer batch size.
        # Provide it explicitly for correct epoch-weighting and to avoid warnings.
        default_batch_size: int = int(next(iter(model_outputs.values())).shape[0]) if model_outputs else 1
        for task in model_outputs:
            task_outputs: torch.Tensor = model_outputs[task]
            metrics = self.task_metrics[task]
            spec = self.task_specs_by_name[task]
            if spec.dataloader_idx is not None and dataloader_idx not in [None, spec.dataloader_idx]:
                continue

            task_batch_size: int = int(task_outputs.shape[0])

            task_batch = batch[task] if isinstance(batch, dict) else batch

            if spec.label_format == 'binary_quantile_dict':
                assert spec.label_quantile is not None
                labels: torch.Tensor = task_batch[1][spec.label_quantile]
                elm_indices: torch.Tensor = task_batch[2]
                task_outputs = task_outputs.reshape_as(labels)
                if spec.track_elmwise_f1 and (
                    (self.current_epoch % self.elmwise_f1_interval == 0 and stage in STAGE_KEYS_FIT)
                    or stage == 'test'
                ):
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
            elif spec.label_format == 'binary_logit':
                labels = task_batch[1]
                task_outputs = task_outputs.reshape_as(labels)
            elif spec.label_format == 'multiclass_index':
                labels = task_batch[1].flatten()
            else:
                raise ValueError(f"Unknown label_format: {spec.label_format}")

            # --- Loss/stat metrics (scalar tensors) ---
            for metric_name, metric_function in metrics.items():
                if 'loss' in metric_name:
                    if metric_name == 'bce_loss':
                        pos_weight = getattr(self, f"loss_pos_weight__{task}", None)
                        pos_weight_t = (
                            pos_weight.to(device=task_outputs.device, dtype=task_outputs.dtype)
                            if isinstance(pos_weight, torch.Tensor)
                            else None
                        )
                        metric_value = torch.nn.functional.binary_cross_entropy_with_logits(
                            input=task_outputs,
                            target=labels.type_as(task_outputs),
                            pos_weight=pos_weight_t,
                        )
                    elif metric_name == 'ce_loss':
                        class_weight = getattr(self, f"loss_class_weight__{task}", None)
                        class_weight_t = (
                            class_weight.to(device=task_outputs.device, dtype=task_outputs.dtype)
                            if isinstance(class_weight, torch.Tensor)
                            else None
                        )
                        metric_value = torch.nn.functional.cross_entropy(
                            input=task_outputs,
                            target=labels,
                            weight=class_weight_t,
                        )
                    else:
                        metric_value = metric_function(
                            input=task_outputs,
                            target=labels.type_as(task_outputs) if spec.task_type == 'binary' else labels,
                        )
                    prod_loss = prod_loss * metric_value
                    task_losses[task] = metric_value
                    if self.multiobjective_method == 'logsigma' and self.is_multitask:
                        metric_weighted = metric_value / (torch.exp(self.task_log_sigma[task])) + self.task_log_sigma[task]
                        sum_loss = sum_loss + metric_weighted
                    else:
                        sum_loss = sum_loss + metric_value
                elif 'stat' in metric_name:
                    metric_value = metric_function(task_outputs)
                else:
                    raise ValueError(
                        f"Unknown metric kind for {metric_name!r} in task {task!r}. "
                        "Expected 'loss' or 'stat' metric names here."
                    )

                # Reduce step-level logging volume: aggregate on epoch.
                self.log(
                    f"{task}/{metric_name}/{stage}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=sync_dist_epoch,
                    add_dataloader_idx=False,
                    batch_size=task_batch_size,
                )

            # --- Score metrics (TorchMetrics) ---
            if stage in STAGE_KEYS_TRAIN_VAL_TEST:
                if spec.task_type == 'binary':
                    preds_for_metrics = torch.sigmoid(task_outputs)
                    target_for_metrics = labels.to(dtype=torch.int)
                else:
                    preds_for_metrics = task_outputs
                    target_for_metrics = labels

                score_metric_names: tuple[str, ...] = tuple(self.task_configs[task].get('score_metric_names', ()))
                for metric_name in score_metric_names:
                    key = f"{task}__{stage}__{metric_name}"
                    metric_obj = cast(Metric, self.stage_task_metrics[key])
                    metric_obj.update(preds_for_metrics, target_for_metrics)

                    # Log the Metric object so Lightning calls compute()/reset() at epoch boundaries.
                    self.log(
                        f"{task}/{metric_name}/{stage}",
                        metric_obj,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=sync_dist_epoch,
                        add_dataloader_idx=False,
                        batch_size=task_batch_size,
                    )

        # Reduce step-level logging volume: aggregate on epoch.
        self.log(
            f"sum_loss/{stage}",
            sum_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist_epoch,
            batch_size=default_batch_size,
        )
        self.log(
            f"prod_loss/{stage}",
            prod_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist_epoch,
            batch_size=default_batch_size,
        )
        return (sum_loss, task_losses) if return_task_losses else sum_loss

    def forward(
            self, 
            batch: dict|list, 
    ) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}

        # Compute trunk features once per input stream (TaskSpec.input_key).
        if isinstance(batch, dict):
            features_by_key: dict[str, torch.Tensor] = {}
            for input_key, tasks in self.input_key_to_tasks.items():
                rep_task = next((t for t in tasks if t in batch), tasks[0])
                x = batch[rep_task][0]
                feats = self.feature_model(x)
                # if self.latent_bn is not None:
                #     feats = self.latent_bn(feats)
                features_by_key[input_key] = feats

            for spec in self.task_specs:
                task_name = spec.name
                feats = features_by_key[self.task_input_key[task_name]]
                results[task_name] = self.task_models[task_name](feats)
        else:
            x = batch[0]
            feats = self.feature_model(x)
            # if self.latent_bn is not None:
            #     feats = self.latent_bn(feats)
            for spec in self.task_specs:
                task_name = spec.name
                results[task_name] = self.task_models[task_name](feats)

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

        # Optional torch.compile (PyTorch 2.x): compile after Lightning has placed the model.
        if (not self._torch_compile_done) and self.use_torch_compile and hasattr(torch, 'compile'):
            # Backend selection:
            # - Allow explicit override via env var TORCH_COMPILE_BACKEND.
            # - Default to eager for maximum robustness on this stack.
            #   (We've observed TorchDynamo backend fake-tensor failures with aot_eager/inductor here.)
            #   Users can opt into other backends via TORCH_COMPILE_BACKEND.
            backend_env = os.getenv('TORCH_COMPILE_BACKEND', '').strip()
            if backend_env:
                # Support a single backend name or a comma-separated preference list.
                backend_candidates: list[str] = [b.strip() for b in backend_env.split(',') if b.strip()]
            else:
                backend_candidates = [
                    # Optional fallbacks (can be faster but are more fragile/noisy).
                    'inductor',
                    'aot_eager',
                    # Most robust choice on this stack, including under TORCH_LOGS/TORCHDYNAMO_VERBOSE.
                    'eager',
                ]

            # Make compile best-effort: if the backend fails at runtime, fall back to eager.
            # NOTE: avoid `import torch._dynamo` here; in a function scope that would bind a local
            # name `torch` and break earlier references.
            try:
                import importlib

                dynamo = importlib.import_module('torch._dynamo')
                dynamo.config.suppress_errors = True
            except Exception:
                pass

            if self.is_global_zero:
                self.zprint(
                    "Compiling trunk with torch.compile "
                    f"(backend candidates={backend_candidates})"
                )

            # TorchDynamo FakeTensor device propagation can fail if module params/buffers are on CPU
            # while inputs are on CUDA. Ensure the trunk is on the same device Lightning will use.
            target_device = getattr(getattr(trainer, 'strategy', None), 'root_device', None) or self.device
            try:
                self.feature_model = self.feature_model.to(target_device)
            except Exception:
                pass

            compiled_any = False
            last_err: Exception | None = None
            for backend in backend_candidates:
                try:
                    original_feature_model = self.feature_model
                    # Compile only the shared trunk. In this codebase/environment we've seen
                    # TorchDynamo fake-tensor backend failures when compiling some heads (BatchNorm1d).
                    # Trunk compilation captures most of the benefit and avoids noisy graph-break warnings.
                    # NOTE: keep fullgraph=False. With fullgraph=True TorchDynamo may raise
                    # DataDependentOutputException (aten._local_scalar_dense) on this stack, especially
                    # when TORCH_LOGS / TORCHDYNAMO_VERBOSE are enabled.
                    self.feature_model = torch.compile(self.feature_model, backend=backend, fullgraph=False)  # type: ignore[attr-defined]

                    # Re-assert device placement after wrapping.
                    try:
                        self.feature_model = self.feature_model.to(target_device)
                    except Exception:
                        pass

                    # Force a warmup call now so backend failures happen here (and can be caught),
                    # rather than crashing at the first real training step.
                    try:
                        dm = getattr(trainer, 'datamodule', None)
                        warmup_bs: int | None = getattr(dm, 'batch_size', None) if dm is not None else None
                        if isinstance(warmup_bs, int) and warmup_bs > 0:
                            per_rank_bs = max(1, int(warmup_bs) // max(1, int(getattr(trainer, 'world_size', 1) or 1)))
                        else:
                            per_rank_bs = 2
                        with torch.no_grad():
                            x = torch.randn((per_rank_bs, *self.input_data_shape[1:]), device=target_device)
                            _ = self.feature_model(x)
                    except Exception as warmup_err:
                        self.feature_model = original_feature_model
                        raise warmup_err

                    if self.is_global_zero:
                        self.zprint(f"torch.compile enabled (backend={backend})")
                    compiled_any = True
                    break
                except Exception as e:
                    last_err = e
                    # Ensure we leave the model in eager mode after a failed attempt.
                    try:
                        self.feature_model = original_feature_model  # type: ignore[has-type]
                    except Exception:
                        pass
                    if self.is_global_zero:
                        self.zprint(f"torch.compile failed (backend={backend}); trying fallback: {e}")

            if not compiled_any and self.is_global_zero and last_err is not None:
                self.zprint(f"torch.compile failed for all backends; continuing without compile: {last_err}")
            self._torch_compile_done = True
        if self.world_size > 0:
            print(f"World rank {self.world_rank} of size {self.world_size} on {self.num_nodes} node(s)")
        if self.num_nodes > 1:
            local_rank = int(os.getenv("SLURM_LOCALID", default=0))
            node_rank = int(os.getenv("SLURM_NODEID", default=0))
            print(f"  Local rank {local_rank} on node {node_rank}")
        trainer.strategy.barrier()

        # Pull loss weighting statistics from the DataModule (computed on train split) and
        # store them as buffers on the model for checkpointing + device moves.
        if stage in (None, 'fit') and getattr(trainer, 'datamodule', None) is not None:
            datamodule = trainer.datamodule
            if getattr(datamodule, 'use_class_imbalance_weights', False):
                pos_w: dict[str, float] = getattr(datamodule, 'task_loss_pos_weight', {}) or {}
                ce_w: dict[str, list[float]] = getattr(datamodule, 'task_loss_class_weight', {}) or {}

                for task_name, v in pos_w.items():
                    # BCEWithLogits pos_weight is a 1D tensor with size = [1] for binary.
                    self.register_buffer(
                        f"loss_pos_weight__{task_name}",
                        torch.tensor([float(v)], dtype=torch.float32),
                        persistent=True,
                    )
                for task_name, wts in ce_w.items():
                    self.register_buffer(
                        f"loss_class_weight__{task_name}",
                        torch.tensor([float(x) for x in wts], dtype=torch.float32),
                        persistent=True,
                    )

    def on_fit_start(self):
        self.t_fit_start = time.time()
        self.zprint(f"**** Fit start with max epochs {self.trainer.max_epochs} ****")
        # for param in self.task_log_sigma.parameters():
        #     param.requires_grad = False

        # Log loss weights once (rank 0) for reproducibility.
        if self.is_global_zero:
            for task_name in self.task_names:
                pw = getattr(self, f"loss_pos_weight__{task_name}", None)
                cw = getattr(self, f"loss_class_weight__{task_name}", None)
                if isinstance(pw, torch.Tensor):
                    self.zprint(f"  loss_pos_weight/{task_name}: {float(pw.detach().cpu().flatten()[0]):.4f}")
                if isinstance(cw, torch.Tensor):
                    self.zprint(f"  loss_class_weight/{task_name}: {cw.detach().cpu().tolist()}")

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        # Step schedulers after validation when monitoring val metrics.
        self._manual_step_lr_schedulers(where='val_epoch_end')

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
        if self.is_multitask and self.current_epoch==self.unfreeze_logsigma_epoch:
            pass
            # self.zprint(f"  Unfreezing task logsigma parameters and adding to optimizer")
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

            def _as_float(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().item()) if x.numel() == 1 else float(x.detach().cpu().mean().item())
                return float(x)

            line =  f"Ep {self.current_epoch:03d}"
            line += f"  bs {self.current_batch_size}"
            line += f" max lr {self.current_max_lr:.1e}"
            line += f" tr/val loss {logged_metrics['sum_loss/train']:.3f}/"
            if 'sum_loss/val' in logged_metrics:
                sum_loss_val = _as_float(logged_metrics['sum_loss/val'])
            else:
                val_keys = sorted(
                    k for k in logged_metrics.keys()
                    if k.startswith('sum_loss/val/dataloader_idx_')
                )
                if val_keys:
                    sum_loss_val = float(np.sum([_as_float(logged_metrics[k]) for k in val_keys]))
                else:
                    sum_loss_val = float('nan')
            line += f"{sum_loss_val:.3f}"
            line += f" ep/gl steps {epoch_steps:,d}/{self.global_step:,d}"
            line += f" ep/gl time (min): {epoch_time/60:.1f}/{global_time/60:.1f}" 
            if self.n_frozen_layers:
                line += f" n_frozen: {self.n_frozen_layers}" 
            print(line)

            if self.multiobjective_method == 'gradnorm' and self.is_multitask:
                w_items: list[str] = []
                for task in self.task_names:
                    w = getattr(self, f"gradnorm_w__{task}", None)
                    if isinstance(w, torch.Tensor):
                        w_items.append(f"{task}={float(w.detach().cpu().flatten()[0]):.3f}")
                if w_items:
                    print("GradNorm w: " + ", ".join(w_items))
        self.save_elm_wise_f1_scores()

        # Step schedulers after training epoch when monitoring train metrics and to
        # advance warmup schedulers.
        self._manual_step_lr_schedulers(where='train_epoch_end')

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
            for stage in STAGE_KEYS_TRAIN_VAL_TEST:
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
        boxcar_window = np.ones(n_boxcar, dtype=float) / float(n_boxcar)
        lambda_smooth = lambda x: np.convolve(x, boxcar_window, mode='valid')
        trainer: Trainer = self.trainer
        dataloaders: Sequence[torch.utils.data.DataLoader] = trainer.predict_dataloaders
        datamodule: Data = trainer.datamodule
        for i_dl, batch_list in self.predict_outputs.items():
            self.rprint(f"  Plotting predictions for dataloader {i_dl}")
            task = datamodule.loader_tasks[i_dl]
            spec = self.task_specs_by_name.get(task)
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
                predictions = _expit_np(outputs)
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
                num_classes = int(spec.num_classes) if spec is not None and spec.num_classes is not None else int(outputs.shape[-1])
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
                smoothed_outputs = np.ndarray((outputs.shape[0]-n_boxcar+1, num_classes), dtype=float)
                for i in range(num_classes):
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
                for i in range(num_classes):
                    # Multi-class: logits -> probabilities via softmax
                    probs = _softmax_np(outputs, axis=1)[:, i]
                    sm_probs = _softmax_np(smoothed_outputs, axis=1)[:, i]
                    plt.plot(times, probs, c=f'C{i}', lw=0.6,)
                    plt.plot(times[n_boxcar-1:], sm_probs, c=f'C{i}', lw=2,)
                plt.xlim(axes[0].get_xlim())
                plt.ylabel('Probability predictions')
                plt.xlabel('Time (ms)')
                if spec is not None and spec.class_labels is not None and len(spec.class_labels) == num_classes:
                    labs = list(spec.class_labels)
                else:
                    labs = [f'class {i}' for i in range(num_classes)]
                rects = []
                for i in range(num_classes):
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
        # Computing/logging grad norms can be expensive; throttle to log cadence.
        log_every_n_steps = getattr(self.trainer, 'log_every_n_steps', None) if self.trainer is not None else None
        if isinstance(log_every_n_steps, int) and log_every_n_steps > 1:
            if (self.global_step % log_every_n_steps) != 0:
                return
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=True, sync_dist=False)

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

    # Task configuration (source of truth). If None, defaults to _default_task_specs().
    task_specs: Sequence[TaskSpec] = None

    # Class imbalance handling
    use_class_imbalance_weights: bool = True
    task_loss_pos_weight: dict[str, float] = dataclasses.field(default_factory=dict)
    task_loss_class_weight: dict[str, list[float]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        _Base_Class.__post_init__(self)
        LightningDataModule.__init__(self)
        # task_specs is owned/logged by the LightningModule to avoid hparams merge conflicts.
        self.save_hyperparameters(ignore=['task_specs'])

        if self.is_global_zero:
            print_fields(self)

        self.trainer: Trainer = None

        self.state_items = ['seed']

        self.allow_zero_length_dataloader_with_multiple_devices = True

        if self.task_specs is None:
            self.task_specs = _default_task_specs()
        self.task_specs = _validate_task_specs(self.task_specs)
        self.has_elm_class = any(spec.name == 'elm_class' for spec in self.task_specs)
        self.has_conf_onehot = any(spec.name == 'conf_onehot' for spec in self.task_specs)

        multiclass_num_classes = {
            int(spec.num_classes)
            for spec in self.task_specs
            if spec.task_type == 'multiclass' and spec.num_classes is not None
        }
        if multiclass_num_classes:
            if len(multiclass_num_classes) != 1:
                raise ValueError(f"Multiple multiclass num_classes in task_specs: {sorted(multiclass_num_classes)}")
            self.num_classes = next(iter(multiclass_num_classes))

        for task_name in (spec.name for spec in self.task_specs):
            assert task_name in ['elm_class', 'conf_onehot'], f"Unknown task {task_name}"

        self.signal_standardization_mean: float = None
        self.signal_standardization_stdev: float = None
        self.state_items.extend([
            'signal_standardization_mean',
            'signal_standardization_stdev',
        ])
        self.elm_sw_count_by_stage: dict[str, int] = {}
        self.conf_sw_count_by_stage: dict[str, int] = {}
        if self.has_elm_class:
            self.elm_data_file = Path(self.elm_data_file).absolute()
            assert self.elm_data_file.exists(), f"ELM data file {self.elm_data_file} does not exist"
            self.global_elm_data_shot_split: dict[str,Sequence] = {}
            self.time_to_elm_quantiles: dict[float,float] = {}
            self.state_items.extend([
                'global_elm_data_shot_split',
                'time_to_elm_quantiles',
            ])
            self.elm_signal_window_metadata: dict[str,Sequence[dict]] = {}
            self.elm_datasets: dict[str,torch.utils.data.Dataset] = {}

        # Class-imbalance weights are computed from the TRAIN split in prepare_data() on rank 0
        # and broadcast/saved for consistent use across ranks.
        self.state_items.extend([
            'task_loss_pos_weight',
            'task_loss_class_weight',
        ])

        if self.has_conf_onehot:
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
        if self.has_elm_class:
            self.prepare_elm_data()
        if self.has_conf_onehot:
            self.prepare_confinement_data()
        self.save_state_dict()
        del self.prepare_rng

    def setup(self, stage: str):
        # called on all ranks after "prepare_data()"
        self.zprint("\u2B1C " + f"Setup stage {stage} (all ranks)")
        # self.rank_rng = np.random.default_rng()
        assert stage in TRAINER_STAGE_KEYS, f"Invalid stage: {stage}"
        assert self.is_global_zero == self.trainer.is_global_zero
        assert self.world_size == self.trainer.world_size
        assert self.world_rank == self.trainer.global_rank
        sub_stages = STAGE_KEYS_FIT if stage == 'fit' else (stage,)
        sub_stages = tuple(_validate_sub_stage_key(s) for s in sub_stages)
        if self.has_elm_class:
            t_tmp = time.time()
            for sub_stage in sub_stages:
                self.setup_elm_data_for_rank(sub_stage)
            self.zprint(f"  ELM data setup time: {time.time()-t_tmp:0.1f} s")
            self.barrier()
        if self.has_conf_onehot:
            t_tmp = time.time()
            for sub_stage in sub_stages:
                self.setup_confinement_data_for_rank(sub_stage)
            self.save_hyperparameters({
                'conf_sw_count_by_stage': self.conf_sw_count_by_stage,
            })
            self.zprint(f"  Confinement data setup time: {time.time()-t_tmp:.1f} s")
            self.barrier()

        # Broadcast computed loss-weighting statistics (train-split class imbalance) to all ranks.
        if self.use_class_imbalance_weights:
            self.task_loss_pos_weight = self.broadcast(self.task_loss_pos_weight)
            self.task_loss_class_weight = self.broadcast(self.task_loss_class_weight)
        self.zprint("\u2B1C Data setup summary")
        for sub_stage in sub_stages:
            self.zprint(f"  Stage {sub_stage}")
            for spec in self.task_specs:
                task = spec.name
                datasets = self.elm_datasets[sub_stage] if task == 'elm_class' else self.confinement_datasets[sub_stage]
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
            # shuffle shots in database
            self.zprint("    Shuffling datafile shots")
            self.prepare_rng.shuffle(datafile_shots)
            # check for reloaded data state
            if self.global_elm_data_shot_split:
                self.zprint("    Global shot split was pre-loaded")
            else:
                self.zprint("    Splitting datafile shots into train/val/test")
                n_test_shots = int(self.fraction_test * len(datafile_shots))
                n_validation_shots = int(self.fraction_validation * len(datafile_shots))
                # split shots in database into train/val/test
                self.global_elm_data_shot_split['test'] = datafile_shots[:n_test_shots]
                self.global_elm_data_shot_split['val'] = datafile_shots[n_test_shots:n_test_shots+n_validation_shots]
                self.global_elm_data_shot_split['train'] = datafile_shots[n_test_shots+n_validation_shots:]
                self.global_elm_data_shot_split['predict'] = self.global_elm_data_shot_split['test']
            # global shot split
            for stage, shotlist in self.global_elm_data_shot_split.items():
                self.zprint(f"      {stage.upper()} shots: {shotlist if len(shotlist)<=7 else shotlist[0:7]}")
            for stage in STAGE_KEYS_FIT:
                assert stage in self.global_elm_data_shot_split and len(self.global_elm_data_shot_split[stage])>0
            # prepare ELMs for stages
            if self.max_elms:
                # Use integer floors for val/test and assign the remainder to train.
                # This avoids surprising zero-TRAIN allocations for tiny `max_elms`.
                n_val = int(self.fraction_validation * self.max_elms)
                n_test = int(self.fraction_test * self.max_elms)
                n_train = int(self.max_elms) - n_val - n_test
                n_elms = {
                    'train': max(0, n_train),
                    'val': max(0, n_val),
                    'test': max(0, n_test),
                }
            # prepare data for stages
            for sub_stage in STAGE_KEYS_TRAIN_VAL_TEST:
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
                if len(global_elms_for_stage) == 0:
                    if sub_stage == 'train':
                        raise ValueError(
                            "No ELMs selected for TRAIN stage. "
                            f"Computed stage allocation from max_elms={self.max_elms}, "
                            f"fraction_validation={self.fraction_validation}, fraction_test={self.fraction_test}: {n_elms}. "
                            "Increase max_elms or adjust split fractions."
                        )
                    self.zprint(f"      No ELMs selected for {sub_stage.upper()} stage; skipping")
                    self.elm_signal_window_metadata[sub_stage] = []
                    if sub_stage == 'test':
                        self.elm_signal_window_metadata['predict'] = []
                    continue
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
                                fsignals = self.apply_fir_filter(signals) if self.b_coeffs is not None else signals
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

                if len(elm_signal_window_metadata) == 0:
                    if sub_stage == 'train':
                        raise ValueError(
                            "TRAIN stage produced zero usable signal windows. "
                            "This can happen if all ELMs are filtered (min_pre_elm_time/outlier_value) or if the data file is too small."
                        )
                    self.zprint(f"      No usable signal windows for {sub_stage.upper()} stage; skipping")
                    self.elm_signal_window_metadata[sub_stage] = []
                    if sub_stage == 'test':
                        self.elm_signal_window_metadata['predict'] = []
                    continue

                # balance signal windows across world_size
                if self.world_size > 1:
                    remainder = len(elm_signal_window_metadata) % self.world_size
                    if remainder:
                        elm_signal_window_metadata = elm_signal_window_metadata[:-remainder]
                assert len(elm_signal_window_metadata) % self.world_size == 0
                self.zprint(f"      Rank-balanced global signal window count: {len(elm_signal_window_metadata):,d}")

                # stats
                if np.sum(cummulative_hist) == 0:
                    # Small datasets may never hit the periodic stats sampling interval.
                    # Compute a single histogram from one representative window as a fallback.
                    pick = elm_signal_window_metadata[len(elm_signal_window_metadata) // 2]
                    elm_index = int(pick['elm_index'])
                    i_t0 = int(pick['i_t0'])
                    elm_event: h5py.Group = root['elms'][f"{elm_index:06d}"]
                    signals = np.array(elm_event["bes_signals"], dtype=np.float32)
                    if self.b_coeffs is not None:
                        signals = self.apply_fir_filter(signals)
                    fsignal_window = signals[..., i_t0:i_t0 + self.signal_window_size]
                    signal_min = np.min([signal_min, fsignal_window.min()])
                    signal_max = np.max([signal_max, fsignal_window.max()])
                    hist, bin_edges = np.histogram(
                        fsignal_window,
                        bins=n_bins,
                        range=(-10.4, 10.4),
                    )
                    cummulative_hist += hist

                bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
                mean = np.sum(cummulative_hist * bin_center) / np.sum(cummulative_hist)
                stdev = np.sqrt(np.sum(cummulative_hist * (bin_center - mean) ** 2) / np.sum(cummulative_hist))
                exkurt = np.sum(cummulative_hist * ((bin_center - mean)/stdev) ** 4) / np.sum(cummulative_hist) - 3
                self.zprint(f"      Signal stats (post-FIR, if used):  mean {mean:.3f}  stdev {stdev:.3f}  exkurt {exkurt:.3f}  min/max {signal_min:.3f}/{signal_max:.3f}")

                # mean/stdev for standarization
                if sub_stage == 'train':
                    self.save_hyperparameters({
                        'elm_raw_signal_mean': mean.item(),
                        'elm_raw_signal_stdev': stdev.item(),
                    })
                    self.zprint("\u2B1C Standardization")
                    if self.signal_standardization_mean:
                        self.zprint(f"  Using pre-loaded mean/stdev for standardization")
                    else:
                        self.zprint(f"  Using {sub_stage.upper()} mean and stdev for standardization")
                        self.signal_standardization_mean = mean.item()
                        self.signal_standardization_stdev = stdev.item()
                    self.zprint(f"  Standarizing signals with mean {self.signal_standardization_mean:.3f} and std {self.signal_standardization_stdev:.3f}")
                    self.save_hyperparameters({
                        'standardization_mean': self.signal_standardization_mean,
                        'standardization_stdev': self.signal_standardization_stdev,
                    })

                    quantiles = [0.5]
                    self.zprint("\u2B1C Time-to-ELM quantiles")
                    if self.time_to_elm_quantiles:
                        self.zprint(f"  Using pre-loaded time-to-ELM quantiles")
                    else:
                        self.zprint(f"  Using {sub_stage.upper()} time-to-ELM quantiles")
                        time_to_elm_labels = [sig_win['time_to_elm'] for sig_win in elm_signal_window_metadata]
                        quantile_values = np.quantile(time_to_elm_labels, quantiles)
                        self.time_to_elm_quantiles = {q: qval.item() for q, qval in zip(quantiles, quantile_values)}
                    for q, qval in self.time_to_elm_quantiles.items():
                        self.zprint(f"  Quantile {q:.2f}: {qval:.2f} ms")
                    self.save_hyperparameters({
                        'time_to_elm_quantiles': self.time_to_elm_quantiles,
                    })

                    # Compute train-split class imbalance weights for binary ELM classification.
                    # Label is defined by time_to_elm <= quantile_threshold.
                    if self.use_class_imbalance_weights:
                        for spec in self.task_specs:
                            if spec.name != 'elm_class':
                                continue
                            if spec.task_type != 'binary':
                                continue
                            if spec.label_format != 'binary_quantile_dict':
                                continue
                            if spec.label_quantile is None:
                                continue
                            q = float(spec.label_quantile)
                            if q not in self.time_to_elm_quantiles:
                                continue
                            thr = float(self.time_to_elm_quantiles[q])
                            time_to_elm_vals = np.array([sig_win['time_to_elm'] for sig_win in elm_signal_window_metadata], dtype=float)
                            pos = int(np.sum(time_to_elm_vals <= thr))
                            neg = int(time_to_elm_vals.size - pos)
                            if pos <= 0 or neg <= 0:
                                self.zprint(f"  WARNING: ELM class imbalance degenerate for q={q}: pos={pos} neg={neg}; skipping pos_weight")
                            else:
                                self.task_loss_pos_weight[spec.name] = float(neg) / float(pos)
                                self.zprint(f"  ELM pos_weight ({spec.name}): {self.task_loss_pos_weight[spec.name]:.3f} (pos={pos} neg={neg})")

                # set quantile limits
                if self.time_to_elm_quantile_min is not None and self.time_to_elm_quantile_max is not None:
                    time_to_elm_labels = np.array([sig_win['time_to_elm'] for sig_win in self.elm_signal_window_metadata])
                    time_to_elm_min, time_to_elm_max = np.quantile(time_to_elm_labels, (self.time_to_elm_quantile_min, self.time_to_elm_quantile_max))
                    # contrastive learning
                    if self.contrastive_learning:
                        self.zprint(f"      Contrastive learning with time-to-ELM quantiles 0.0-{self.time_to_elm_quantile_min:.2f} and {self.time_to_elm_quantile_max:.2f}-1.0")
                        for i in np.arange(len(elm_signal_window_metadata)-1, -1, -1, dtype=int):
                            if (elm_signal_window_metadata[i]['time_to_elm'] > time_to_elm_min) and \
                                (elm_signal_window_metadata[i]['time_to_elm'] < time_to_elm_max):
                                elm_signal_window_metadata.pop(i)
                    else:
                        self.zprint(f"      Restricting time-to-ELM labels to quantile range: {self.time_to_elm_quantile_min:.2f}-{self.time_to_elm_quantile_max:.2f}")
                        for i in np.arange(len(elm_signal_window_metadata)-1, -1, -1, dtype=int):
                            if (elm_signal_window_metadata[i]['time_to_elm'] < time_to_elm_min) or \
                                (elm_signal_window_metadata[i]['time_to_elm'] > time_to_elm_max):
                                elm_signal_window_metadata.pop(i)

                # final global ELM metadata for stage
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
        sub_stage = _validate_sub_stage_key(sub_stage)
        self.zprint("  \u2B1C " + f"Setup ELM data for stage {sub_stage.upper()}")
        if sub_stage == 'predict' and 'predict' in self.elm_datasets and self.elm_datasets['predict']:
            self.zprint("    ELM `predict` dataset already setup")
            return
        # broadcast global ELM data specifications
        self.elm_signal_window_metadata = self.broadcast(self.elm_signal_window_metadata)
        # broadcast signal standardization
        self.signal_standardization_mean = self.broadcast(self.signal_standardization_mean)
        self.signal_standardization_stdev = self.broadcast(self.signal_standardization_stdev)
        # broadcast time-to-ELM quantiles for labeling
        self.time_to_elm_quantiles = self.broadcast(self.time_to_elm_quantiles)

        # Empty stages are valid (e.g., tiny max_elms + fractional splits).
        # Create empty datasets and return early.
        sw_list_global = self.elm_signal_window_metadata.get(sub_stage, [])
        if not sw_list_global:
            self.zprint(f"    No ELM signal windows for {sub_stage.upper()} stage")
            if sub_stage in STAGE_KEYS_TRAIN_VAL_TEST:
                self.elm_datasets[sub_stage] = ELM_TrainValTest_Dataset(
                    signal_window_size=self.signal_window_size,
                    time_to_elm_quantiles=self.time_to_elm_quantiles or {0.5: 0.0},
                    sw_list=[],
                    elms_to_signals_dict={},
                )
            if sub_stage in ('test', 'predict'):
                self.elm_datasets['predict'] = []
            return

        # get rank-wise ELM signals
        sw_for_stage: np.ndarray = np.array(sw_list_global, dtype=object)
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
                    signals = self.apply_fir_filter(signals)
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
        if sub_stage in STAGE_KEYS_TRAIN_VAL_TEST:
            self.elm_datasets[sub_stage] = ELM_TrainValTest_Dataset(
                signal_window_size=self.signal_window_size,
                time_to_elm_quantiles=self.time_to_elm_quantiles,
                sw_list=sw_for_rank,
                elms_to_signals_dict=elms_to_signals_map,
            )
        if sub_stage in ('test', 'predict'):
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
        stages = list(STAGE_KEYS_TRAIN_VAL_TEST)
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
        train_class_to_sigwins: Optional[list[int]] = None
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

            if st == 'train':
                train_class_to_sigwins = list(class_to_sigwins)

        # Compute train-split class weights for multiclass confinement.
        if self.use_class_imbalance_weights and train_class_to_sigwins is not None:
            total = float(sum(train_class_to_sigwins))
            if total > 0 and all(c > 0 for c in train_class_to_sigwins):
                weights = [total / (self.num_classes * float(c)) for c in train_class_to_sigwins]
                for spec in self.task_specs:
                    if spec.name == 'conf_onehot' and spec.task_type == 'multiclass':
                        self.task_loss_class_weight[spec.name] = [float(w) for w in weights]
                        self.zprint(f"    Conf class weights ({spec.name}): {self.task_loss_class_weight[spec.name]}")
            else:
                self.zprint(f"    WARNING: Cannot compute conf class weights; counts={train_class_to_sigwins}")

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
        sub_stage = _validate_sub_stage_key(sub_stage)
        self.zprint("  \u2B1C " + f"Setup confinement data for stage {sub_stage.capitalize()}")
        if sub_stage == 'predict' and 'predict' in self.confinement_datasets and self.confinement_datasets['predict']:
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
                signals_raw = np.array(event_group["signals"][:, :max_length], dtype=np.float32)
                bes_time = np.array(event_group["time"][:max_length], dtype=np.float32)  # (<time>)
                shotevent_to_time_map[shotevent_label] = bes_time
                shotevent_to_shot_and_event_map[shotevent_label] = (int(shot), int(event))
                assert signals_raw.shape[0] == 64
                assert labels.size == signals_raw.shape[1]

                # For outlier detection and windowing logic we use time-first representation.
                # (T,64) -> (T,8,8)
                signals_time_first = np.transpose(signals_raw, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
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
                        if np.max(np.abs(signals_time_first[ii-self.signal_window_size+1:ii+1, ...])) > self.outlier_value:
                            outlier_count += 1
                            valid_t0[ii] = 0
                # FIR filter, if used. Filter along time on the original (64, T) array,
                # then reshape into (T, 8, 8) for downstream.
                if self.b_coeffs is not None:
                    signals_f = self.apply_fir_filter(signals_raw)
                    signals = np.transpose(signals_f, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                else:
                    signals = signals_time_first
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
        if self.trainer.world_size > 1 and sub_stage in STAGE_KEYS_FIT:
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
            sub_stage in STAGE_KEYS_FIT:
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

        if sub_stage in STAGE_KEYS_TRAIN_VAL_TEST:
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
        if sub_stage in ('test', 'predict'):
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
        stage = _validate_sub_stage_key(stage)
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

        # This DataModule already partitions datasets per rank in setup_*_for_rank().
        # Avoid DistributedSampler(num_replicas=1, rank=0) overhead.
        shuffle = (stage == 'train')
        generator = None
        if shuffle:
            generator = torch.Generator(device='cpu')
            base_seed = int(self.seed) if self.seed is not None else 0
            epoch = int(getattr(getattr(self, 'trainer', None), 'current_epoch', 0))
            generator.manual_seed((base_seed + epoch) % (2**63 - 1))

        return torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            generator=generator,
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
            self.b_coeffs = _firwin_windowed_sinc(
                numtaps=501,  # must be odd
                cutoff=cutoff,  # in kHz
                pass_zero=pass_zero,
                fs=1e3,  # f_sample in kHz
            )
            self.a_coeffs = None
        else:
            self.zprint("Using raw BES signals; no FIR filter")

    def apply_fir_filter(self, signals: np.ndarray) -> np.ndarray:
        b = getattr(self, 'b_coeffs', None)
        if b is None:
            return np.asarray(signals, dtype=np.float32)

        # Fast causal FIR using grouped torch.conv1d on CPU.
        # Supports:
        # - ELM signals: (64, T)
        # - Confinement signals: (T, 8, 8)
        x_np = np.asarray(signals, dtype=np.float32)
        if x_np.size == 0:
            return x_np.astype(np.float32, copy=False)

        b_np = np.asarray(b, dtype=np.float32).reshape(-1)
        k = int(b_np.size)
        if k <= 1:
            return x_np.astype(np.float32, copy=False)

        with torch.no_grad():
            b_t = torch.from_numpy(b_np)
            # conv1d is cross-correlation, so reverse the kernel for causal FIR.
            w1 = b_t.flip(0).view(1, 1, k)  # (1,1,K)

            if x_np.ndim == 2:
                x_t = torch.from_numpy(x_np)
                # Heuristic: treat (C,T) if C is 64; else treat (T,C).
                if x_t.shape[0] == 64:
                    x_ct = x_t  # (C,T)
                else:
                    x_ct = x_t.transpose(0, 1)
                x_ct = x_ct.unsqueeze(0)  # (1,C,T)
                c = int(x_ct.shape[1])
                w = w1.expand(c, 1, k).contiguous()  # (C,1,K)
                x_pad = torch.nn.functional.pad(x_ct, (k - 1, 0))
                y = torch.nn.functional.conv1d(x_pad, w, groups=c)
                y_ct = y.squeeze(0)
                y_out = y_ct if x_t.shape[0] == 64 else y_ct.transpose(0, 1)
                return y_out.cpu().numpy().astype(np.float32, copy=False)

            if x_np.ndim == 3:
                # Assume time-first (T,H,W) and filter along time.
                t, h, w_ = x_np.shape
                x_t = torch.from_numpy(x_np.reshape(t, h * w_).T)  # (C,T)
                x_t = x_t.unsqueeze(0)  # (1,C,T)
                c = int(x_t.shape[1])
                w = w1.expand(c, 1, k).contiguous()
                x_pad = torch.nn.functional.pad(x_t, (k - 1, 0))
                y = torch.nn.functional.conv1d(x_pad, w, groups=c).squeeze(0)  # (C,T)
                y_t = y.T.reshape(t, h, w_)
                return y_t.cpu().numpy().astype(np.float32, copy=False)

        # Fallback for unexpected shapes.
        return _lfilter_fir_causal(x_np, b_np)

    def train_dataloader(self) -> CombinedLoader:
        loaders = {}
        for spec in self.task_specs:
            if spec.name == 'elm_class':
                loaders['elm_class'] = self.get_dataloader_from_dataset('train', self.elm_datasets['train'])
            elif spec.name == 'conf_onehot':
                loaders['conf_onehot'] = self.get_dataloader_from_dataset('train', self.confinement_datasets['train'])
        combined_loader: CombinedLoader = CombinedLoader(
            iterables=loaders,
            # Define an epoch by the smaller stream to avoid oversampling the other stream.
            mode='min_size',
        )
        _ = iter(combined_loader)
        return combined_loader

    def val_dataloader(self) -> CombinedLoader:
        loaders = {}
        for spec in self.task_specs:
            if spec.name == 'elm_class':
                loaders['elm_class'] = self.get_dataloader_from_dataset('val', self.elm_datasets['val'])
            elif spec.name == 'conf_onehot':
                loaders['conf_onehot'] = self.get_dataloader_from_dataset('val', self.confinement_datasets['val'])
        combined_loader: CombinedLoader = CombinedLoader(
            iterables=loaders,
            mode='sequential',
        )
        _ = iter(combined_loader)
        return combined_loader

    def test_dataloader(self) -> CombinedLoader:
        loaders = {}
        for spec in self.task_specs:
            if spec.name == 'elm_class':
                loaders['elm_class'] = self.get_dataloader_from_dataset('test', self.elm_datasets['test'])
            elif spec.name == 'conf_onehot':
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
        for spec in self.task_specs:
            task = spec.name
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

    def _normalize_stage_keys(self) -> None:
        def _rename_validation_key(d: dict) -> None:
            if isinstance(d, dict) and ('validation' in d) and ('val' not in d):
                d['val'] = d.pop('validation')

        for attr_name in [
            'global_elm_data_shot_split',
            'elm_signal_window_metadata',
            'elm_datasets',
            'elm_sw_count_by_stage',
            'global_conf_data_shot_split',
            'stage_to_rank_to_event_mapping',
            'confinement_datasets',
            'conf_sw_count_by_stage',
        ]:
            if hasattr(self, attr_name):
                _rename_validation_key(getattr(self, attr_name))

    def load_state_dict(self, state: dict) -> None:
        self.zprint("\u2B1C Loading state dict")
        for item in self.state_items:
            self.zprint(f"    {item}: {state[item]}")
            setattr(self, item, state[item])

        # Back-compat: older state dicts used 'validation' keys.
        self._normalize_stage_keys()

    def save_state_dict(self):
        if self.is_global_zero:
            state_dict_file = Path(self.log_dir)/'state_dict.pt'
            state_dict_file.parent.mkdir(parents=True, exist_ok=True)
            self.zprint("\u2B1C " + f"Saving state_dict: {state_dict_file}")
            state_dict = {item: getattr(self, item) for item in self.state_items}
            for key in state_dict:
                self.zprint(f"    {key}:  {state_dict[key]}")
            torch.save(state_dict, state_dict_file)

    def rprint(self, text: str = ''):
        self.barrier()
        super().rprint(text)
        self.barrier()

    def barrier(self):
        if self.world_size <= 1:
            return

        trainer = getattr(self, 'trainer', None)
        strategy = getattr(trainer, 'strategy', None) if trainer is not None else None
        barrier_fn = getattr(strategy, 'barrier', None)
        if callable(barrier_fn):
            barrier_fn()

    def broadcast(self, obj, src: int = 0):
        if self.world_size <= 1:
            return obj

        trainer = getattr(self, 'trainer', None)
        strategy = getattr(trainer, 'strategy', None) if trainer is not None else None
        broadcast_fn = getattr(strategy, 'broadcast', None)
        if callable(broadcast_fn):
            obj = broadcast_fn(obj, src=src)
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
        # batch_norm: bool = True,
        dropout: float = 0.05,
        feature_model_layers: Sequence[dict[str, LightningModule]] = None,
        task_specs: Optional[Sequence[TaskSpec]] = None,
        multiobjective_method: Literal['logsigma', 'pcgrad', 'gradnorm'] = 'logsigma',
        unfreeze_logsigma_epoch: int = -1,
        logsigma_warmup_epochs: int = 0,
        grad_update_interval: int = 1,
        gradnorm_rep_params: str = 'last_trunk_layer',
        use_torch_compile: bool = False,
        # latent_batch_norm: bool = True,
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
        seed: int = None,
        balance_confinement_data_with_elm_data: bool = False,
        bad_elm_indices: Sequence[int] = (),
) -> dict:

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    ### SLURM/MPI environment
    num_nodes = int(os.getenv('SLURM_NNODES', default=1))
    world_size = int(os.getenv("SLURM_NTASKS", default=1))
    world_rank = int(os.getenv("SLURM_PROCID", default=0))
    if world_size > 1:
        print(f"World rank {world_rank} of size {world_size} on {num_nodes} node(s)")

    is_global_zero = (world_rank == 0)
    def zprint(text):
        if is_global_zero:
            print(text)

    if multiobjective_method in ('pcgrad', 'gradnorm') and gradient_clip_val not in (0, None):
        zprint(
            f"multiobjective_method={multiobjective_method} uses manual optimization; "
            f"disabling Lightning gradient clipping (gradient_clip_val={gradient_clip_val} -> 0)."
        )
        gradient_clip_val = 0

    # Guardrail: very small `max_elms` often yields empty VAL (and sometimes empty TRAIN)
    # due to integer rounding, which then breaks callbacks expecting monitored val metrics.
    if max_elms is not None and max_elms < 10:
        zprint(f"Requested max_elms={max_elms} < 10; clamping to max_elms=10")
        max_elms = 10

    # --- Deterministic seeding (torch + numpy + python + dataloader workers) ---
    # In distributed runs, every rank must use the same seed.
    if seed is None:
        seed = np.random.default_rng().integers(0, 2**32 - 1)
    seed = int(seed)
    zprint(f"Using seed: {seed}")
    seed_everything(seed, workers=True)

    ### model
    zprint("\u2B1C Creating model")
    lit_model = Model(
        signal_window_size=signal_window_size,
        no_bias=no_bias,
        # batch_norm=batch_norm,
        feature_model_layers=feature_model_layers,
        task_specs=task_specs,
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
        unfreeze_logsigma_epoch=unfreeze_logsigma_epoch,
        logsigma_warmup_epochs=logsigma_warmup_epochs,
        multiobjective_method=multiobjective_method,
        # latent_batch_norm=latent_batch_norm,
        grad_update_interval=grad_update_interval,
        gradnorm_rep_params=gradnorm_rep_params,
        use_torch_compile=use_torch_compile,
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
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "use_wandb=True requires the 'wandb' package. Install it (e.g. `pip install wandb`) "
                "or set use_wandb=False."
            ) from e
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
        # if ckpt_path:
        #     assert seed is not None
        # TODO: after instantiating datamodule, load state_dict of checkpoint into datamodule
        lit_datamodule = Data(
            signal_window_size=signal_window_size,
            log_dir=trainer.log_dir,
            elm_data_file=elm_data_file,
            confinement_data_file=confinement_data_file,
            task_specs=lit_model.task_specs,
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
        if backbone_model_path:
            state_dict_file = Path(backbone_model_path) / 'state_dict.pt'
            assert state_dict_file.exists(), f"State dict file does not exist: {state_dict_file}"
            print(f"Loading state_dict from: {state_dict_file}")
            state_dict = torch.load(state_dict_file, weights_only=False)
            lit_datamodule.load_state_dict(state_dict)

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

    feature_model_layers = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        # {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
    )

    task_specs = (
        TaskSpec(
            name='elm_class',
            task_type='binary',
            head_layers=(16,),
            label_format='binary_quantile_dict',
            label_quantile=0.5,
            dataloader_idx=0,
            track_elmwise_f1=True,
        ),
        TaskSpec(
            name='conf_onehot',
            task_type='multiclass',
            head_layers=(16,),
            num_classes=4,
            label_format='multiclass_index',
            dataloader_idx=1,
            class_labels=('L-mode','H-mode','QH-mode','WP QH-mode'),
        ),
    )
    main(
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        # elm_data_file=ml_data.small_data_100,
        feature_model_layers=feature_model_layers,
        task_specs=task_specs,
        max_elms=60,
        max_epochs=2,
        # bad_elm_indices=bad_elm_indices,
        lr=1e-2,
        lr_warmup_epochs=2,
        batch_size=128,
        fraction_validation=0.125,
        # fraction_test=0.125,
        num_workers=2,
        max_confinement_event_length=int(10e3),
        confinement_dataset_factor=0.2,
        multiobjective_method='logsigma',
        use_torch_compile=True,
        # use_wandb=True,
        monitor_metric='sum_loss/train',
        # seed=int(np.random.default_rng().integers(0, 2**32-1)),
        # balance_confinement_data_with_elm_data=True,
        # skip_train=True,
        skip_test=True,
        # backbone_model_path='experiment_default/r2025_12_23_11_33_20',
        # backbone_unfreeze_at_epoch=1,
        # backbone_first_n_layers=3,
        # backbone_initial_lr=1e-3,
        # backbone_warmup_rate=2,
    )
