from __future__ import annotations
import dataclasses
from pathlib import Path
import gc
import os
import time
from typing import Iterable
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import firwin, filtfilt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from scipy.stats import mode

import h5py

import torch
import torch.nn
import torch.utils.data
from torch.utils.data import ConcatDataset
import time

from lightning.pytorch import LightningDataModule

from bes_data.sample_data import sample_elm_data_file

import psutil
from collections import defaultdict


class Confinement_TrainValTest_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            signals: np.ndarray,
            n_rows: int,
            n_cols: int,
            labels: np.ndarray,
            sample_indices: np.ndarray,
            window_start_indices: np.ndarray,
            signal_window_size: int,
            confinement_mode_keys: np.ndarray,
    ) -> None:
        # Create a contiguous copy of the array and then convert it to a PyTorch tensor
        self.signals = torch.from_numpy(np.ascontiguousarray(signals)[np.newaxis, ...])
        assert (
            self.signals.ndim == 4 and
            self.signals.size(0) == 1 and
            self.signals.size(2) == n_rows and
            self.signals.size(3) == n_cols
        ), "Signals have incorrect shape"
        self.labels = torch.from_numpy(labels)
        # assert self.labels.ndim == 1, "Labels have incorrect shape"
        print(signals.shape, labels.shape)
        # assert self.labels.numel() == self.signals.size(1), "Labels and signals have different time dimensions"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.signal_window_size = signal_window_size
        self.window_start_indices = torch.from_numpy(window_start_indices)
        self.sample_indices = torch.from_numpy(sample_indices)
        self.confinement_mode_keys = confinement_mode_keys
        assert torch.max(self.sample_indices) < self.labels.shape[0]


        # Create a dictionary to map confinement_mode_keys to start and end indices
        self.confinement_mode_id_to_indices = {}
        for idx, key in enumerate(self.confinement_mode_keys):
            start_idx = self.window_start_indices[idx]
            end_idx = self.window_start_indices[idx + 1] if idx + 1 < len(self.window_start_indices) else self.signals.size(1)
            self.confinement_mode_id_to_indices[key] = (start_idx, end_idx)

    def __len__(self) -> int:
        return self.sample_indices.numel()
    
    def get_full_signal_by_id(self, confinement_mode_id):    
        # Make sure to check what type of key is stored in confinement_mode_id_to_indices
        start_idx, end_idx = self.confinement_mode_id_to_indices.get(confinement_mode_id, (None, None))

        if start_idx is None or end_idx is None:
            print(f"Warning: No indices found for confinement_mode_id: {confinement_mode_id}")
            return None
        if self.n_rows >= 3:
            row_idx = 2
        else:
            row_idx = self.n_rows-1

        if self.n_cols >= 4:
            col_idx = 3
        else:
            col_idx = self.n_cols-1

        print(f"Found start index {start_idx}, end index {end_idx} for confinement_mode_id: {confinement_mode_id}")
        return self.signals[:, start_idx:end_idx, row_idx, col_idx].squeeze(0)

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
        confinement_mode_idx = (self.window_start_indices <= i_t0).nonzero().max()
        confinement_mode_key = self.confinement_mode_keys[confinement_mode_idx]

        # Convert the key to an integer by removing non-numeric characters and converting to int
        confinement_mode_id_int = int(confinement_mode_key.replace('/', ''))

        # Convert to tensor
        confinement_mode_id_tensor = torch.tensor([confinement_mode_id_int], dtype=torch.int64)
        
        return signal_window, label, confinement_mode_id_tensor
    

# class Confinement_Predict_Dataset(torch.utils.data.Dataset):

#     def __init__(
#             self,
#             signals: np.ndarray,
#             labels: np.ndarray,
#             signal_window_size: int,
#             shot: int,
#             start_time: int,
#             confinement_mode_index: int,
#     ) -> None:
#         self.shot = shot
#         self.start_time = start_time
#         self.confinement_mode_index = confinement_mode_index
#         self.signals = torch.from_numpy(signals[np.newaxis, ...])
#         assert (
#             self.signals.ndim == 4 and
#             self.signals.size(0) == 1 and
#             self.signals.size(2) == 2 and
#             self.signals.size(3) == 8
#         ), "Signals have incorrect shape"
#         self.labels = torch.from_numpy(labels)
#         assert self.labels.ndim == 1, "Labels have incorrect shape"
#         assert self.labels.numel() == self.signals.size(1), "Labels and signals have different time dimensions"
#         self.signal_window_size = signal_window_size
#         last_signal_window_start_index = self.labels.numel() - self.signal_window_size
#         assert last_signal_window_start_index+self.signal_window_size == self.labels.numel()
#         valid_t0 = np.zeros(self.labels.numel(), dtype=int)  
#         valid_t0[:last_signal_window_start_index+1] = 1
#         assert valid_t0[last_signal_window_start_index] == 1  # last signal window start 
#         assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start 
#         sample_indices = np.arange(valid_t0.size, dtype=int)
#         sample_indices = sample_indices[valid_t0 == 1]
#         self.sample_indices = torch.from_numpy(sample_indices)

#     def pre_elm_stats(self) -> dict[str, torch.Tensor]:
#         pre_elm_signals = self.signals[0,:self.active_elm_start_index,...]
#         maxabs, _ = torch.max(torch.abs(pre_elm_signals), dim=0)
#         std, mean = torch.std_mean(pre_elm_signals, dim=0)
#         return {
#             'maxabs': maxabs.numpy(force=True),
#             'mean': mean.numpy(force=True),
#             'std': std.numpy(force=True),
#         }

#     def __len__(self) -> int:
#         return self.sample_indices.numel()

#     def __getitem__(self, i: int) -> tuple:
#         i_t0 = self.sample_indices[i]
#         signal_window = self.signals[:, i_t0 : i_t0 + self.signal_window_size, :, :]
#         label_index = i_t0 + self.signal_window_size - 1
#         label = self.labels[ label_index : label_index + 1 ]
#         # label_class = torch.tensor([0]) if label >= 0 else torch.tensor([1])
#         return signal_window, label, self.shot, self.start_time

# class Confinement_Predict_Dataset(torch.utils.data.Dataset):
#     """
#     Expects `signals` as either (T, n_rows, n_cols) or (1, T, n_rows, n_cols).
#     Internally stores as (1, T, n_rows, n_cols) float32.
#     Labels are 1D of length T (float; NaN allowed). We return int64 labels
#     at the window end (NaN -> -1) for CE-style evaluation.
#     """
#     def __init__(self,
#                  signals: np.ndarray | torch.Tensor,
#                  n_rows: int,
#                  n_cols: int,
#                  labels: np.ndarray | torch.Tensor,
#                  signal_window_size: int,
#                  shot: int,
#                  start_time: int,
#                  confinement_mode_index: int):

#         self.shot = int(shot)
#         self.start_time = int(start_time)
#         self.confinement_mode_index = int(confinement_mode_index)
#         self.n_rows = int(n_rows)
#         self.n_cols = int(n_cols)

#         # ---- normalize signals to (1, T, n_rows, n_cols) ----
#         if isinstance(signals, torch.Tensor):
#             sig = signals.detach().cpu().numpy()
#         else:
#             sig = np.asarray(signals)

#         if sig.ndim == 3:
#             # (T, R, C)
#             T, R, C = sig.shape
#             assert R == self.n_rows and C == self.n_cols, \
#                 f"Signals wrong shape: expected (T,{self.n_rows},{self.n_cols}) got {sig.shape}"
#             sig = sig[None, ...]  # -> (1, T, R, C)
#         elif sig.ndim == 4:
#             # (1, T, R, C)
#             assert sig.shape[0] == 1, f"First dim must be channel=1, got {sig.shape[0]}"
#             assert sig.shape[2] == self.n_rows and sig.shape[3] == self.n_cols, \
#                 f"Signals wrong shape: expected (*,*,{self.n_rows},{self.n_cols}) got {sig.shape}"
#         else:
#             raise AssertionError(f"Signals wrong ndim: expected 3 or 4, got {sig.ndim}")

#         self.signals = torch.from_numpy(np.ascontiguousarray(sig)).float()  # (1, T, R, C)

#         # ---- labels ----
#         if isinstance(labels, torch.Tensor):
#             lab = labels.detach().cpu().numpy()
#         else:
#             lab = np.asarray(labels)

#         assert lab.ndim == 1, "Labels must be 1D (length T)"
#         T_total = self.signals.shape[1]
#         assert lab.shape[0] == T_total, \
#             f"Labels/signals time mismatch: labels={lab.shape[0]} vs signals.T={T_total}"
#         self.labels = torch.from_numpy(lab.astype(np.float32))  # keep float; we cast to int at __getitem__

#         # ---- windowing ----
#         self.signal_window_size = int(signal_window_size)
#         assert self.signal_window_size > 0, "signal_window_size must be > 0"
#         last_start = T_total - self.signal_window_size
#         assert last_start >= 0, \
#             f"Window longer than sequence: W={self.signal_window_size} > T={T_total}"

#         # valid window starts: [0, ..., last_start]
#         self.sample_indices = torch.arange(0, last_start + 1, dtype=torch.int64)

#     def __len__(self) -> int:
#         return int(self.sample_indices.numel())

#     def __getitem__(self, i: int) -> tuple:
#         i_t0 = int(self.sample_indices[i])
#         W = self.signal_window_size

#         # slice: (1, W, R, C) — DO NOT unsqueeze again
#         win = self.signals[:, i_t0:i_t0 + W, :, :].contiguous()

#         # label at window end
#         label_idx = i_t0 + W - 1
#         label_val = float(self.labels[label_idx].item())
#         if np.isnan(label_val):
#             label_val = -1
#         label = torch.tensor(int(label_val), dtype=torch.long)

#         return win, label, self.shot, self.start_time


class Confinement_Predict_Dataset(torch.utils.data.Dataset):
    """
    Expects `signals` as either (T, n_rows, n_cols) or (1, T, n_rows, n_cols).
    Internally stores as (1, T, n_rows, n_cols) float32.
    Labels are 1D of length T (float; NaN allowed). We return int64 labels
    at the window end (NaN -> -1) for CE-style evaluation.
    """
    def __init__(self,
                 signals,
                 n_rows: int,
                 n_cols: int,
                 labels,
                 signal_window_size: int,
                 shot: int,
                 start_time: int,
                 confinement_mode_index: int,
                 sample_indices: torch.Tensor | None = None,   # <--- NEW
                 dt_ms: float | None = None):                  # <--- NEW

        self.shot = int(shot)
        # IMPORTANT semantics change: start_time now means "time at the FIRST window end (ms)"
        self.start_time = int(start_time)
        self.confinement_mode_index = int(confinement_mode_index)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.dt_ms = float(dt_ms) if dt_ms is not None else 1.0  # safe default

        # ---- normalize signals to (1, T, n_rows, n_cols) ----
        if isinstance(signals, torch.Tensor):
            sig = signals.detach().cpu().numpy()
        else:
            sig = np.asarray(signals)

        if sig.ndim == 3:
            T, R, C = sig.shape
            assert R == self.n_rows and C == self.n_cols, \
                f"Signals wrong shape: expected (T,{self.n_rows},{self.n_cols}) got {sig.shape}"
            sig = sig[None, ...]  # -> (1, T, R, C)
        elif sig.ndim == 4:
            assert sig.shape[0] == 1, f"First dim must be channel=1, got {sig.shape[0]}"
            assert sig.shape[2] == self.n_rows and sig.shape[3] == self.n_cols, \
                f"Signals wrong shape: expected (*,*,{self.n_rows},{self.n_cols}) got {sig.shape}"
        else:
            raise AssertionError(f"Signals wrong ndim: expected 3 or 4, got {sig.ndim}")

        self.signals = torch.from_numpy(np.ascontiguousarray(sig)).float()  # (1, T, R, C)

        # ---- labels ----
        if isinstance(labels, torch.Tensor):
            lab = labels.detach().cpu().numpy()
        else:
            lab = np.asarray(labels)

        assert lab.ndim == 1, "Labels must be 1D (length T)"
        T_total = self.signals.shape[1]
        assert lab.shape[0] == T_total, \
            f"Labels/signals time mismatch: labels={lab.shape[0]} vs signals.T={T_total}"
        self.labels = torch.from_numpy(lab.astype(np.float32))

        # ---- windowing ----
        self.signal_window_size = int(signal_window_size)
        assert self.signal_window_size > 0, "signal_window_size must be > 0"
        last_start = T_total - self.signal_window_size
        assert last_start >= 0, \
            f"Window longer than sequence: W={self.signal_window_size} > T={T_total}"

        if sample_indices is None:
            # old behavior: fully overlapping stride-1
            self.sample_indices = torch.arange(0, last_start + 1, dtype=torch.int64)
        else:
            self.sample_indices = sample_indices.to(torch.int64)
            # sanity: all starts valid
            assert torch.all((self.sample_indices >= 0) & (self.sample_indices <= last_start)), \
                "sample_indices contains out-of-range starts"

    def __len__(self) -> int:
        return int(self.sample_indices.numel())

    def __getitem__(self, i: int) -> tuple:
        i_t0 = int(self.sample_indices[i])
        W = self.signal_window_size

        # slice: (1, W, R, C)
        win = self.signals[:, i_t0:i_t0 + W, :, :].contiguous()

        # label at window end
        label_idx = i_t0 + W - 1
        label_val = float(self.labels[label_idx].item())
        if np.isnan(label_val):
            label_val = -1
        label = torch.tensor(int(label_val), dtype=torch.long)

        # return the TRUE time at the window end (ms)
        # NOTE: self.start_time == time at the FIRST window end (i_t0 == 0)
        # For a generic i_t0, advance by i_t0 * dt_ms
        time_end_ms = int(round(self.start_time + i_t0 * self.dt_ms))

        return win, label, self.shot, time_end_ms


@dataclasses.dataclass(eq=False)
class Confinement_Datamodule(LightningDataModule):
    data_file: str = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/6x8_confinement_data_8.hdf5'  # path to data; dir or file depending on task
    n_rows: int = 6
    n_cols: int = 8
    batch_size: int = 128  # power of 2, like 32-256
    signal_window_size: int = 128  # power of 2, like 64-512
    predict_window_stride: int = 512
    num_workers: int = 0  # number of subprocess workers for pytorch dataloader
    fraction_validation: float = 0.2  # fraction of dataset for validation
    fraction_test: float = 0.2  # fraction of dataset for testing
    test_only: bool = False
    seed: int = 0  # RNG seed for deterministic, reproducible shuffling of events
    metadata_bounds = {
        'r_avg': None,
        'z_avg': None,
        'delz_avg': None
    }    
    max_shots: int = None
    max_shots_per_class: int = None
    num_classes: int = 4
    max_predict_confinement_modes: int = 24
    bad_shots: list = None
    force_validation_shots: list = None
    force_test_shots: list = None
    predict_shots: list = None
    # Bandpass filter parameters
    sampling_frequency_hz: float = 1 / 10**(-6)  # Sampling frequency in Hz
    target_sampling_hz: float = None
    filter_taps: int = 501  # Number of taps in the filter
    lower_cutoff_frequency_hz: float = None  # Lower cutoff frequency in Hz
    upper_cutoff_frequency_hz: float = None  # Upper cutoff frequency in Hz
    standardize_signals: bool = True
    mask_sigma_outliers: float = None  # remove signal windows with abs(standardized_signals) > n_sigma
    label_filter:  Iterable[int] | str | None = None
    one_hot_labels: bool = False # if True, use one-hot vector for label
    prepare_data_per_node: bool = True  # hack to avoid error between dataclass and LightningDataModule
    plot_data_stats: bool = True
    is_global_zero: bool = dataclasses.field(default=True, init=False)
    log_dir: str = dataclasses.field(default='.', init=False)
    world_size: int = 1 # number of total GPUs 

    def __post_init__(self):
        super().__init__()
        if self.data_file is None:
            self.data_file = sample_elm_data_file.as_posix()
        self.save_hyperparameters(
            ignore=['max_predict_elms']
        )

        # datamodule state, to reproduce pre-processing
        self.state_items = [
            'mask_ub',
            'mask_lb',
            'signal_mean',
            'signal_stdev',
            'signal_exkurt',
            'label_median',
        ]
        for item in self.state_items:
            if not hasattr(self, item):
                setattr(self, item, None)

        self.datasets = {}
        self.all_confinement_events = None
        self.test_confinement_events = None
        self.train_confinement_events = None
        self.validation_confinement_events = None
        self.predict_confinement_events = None
        self._get_confinement_events_and_split()
        self.dataset_confinement_events = {
                'train': self.train_confinement_events,
                'validation': self.validation_confinement_events,
                'test': self.test_confinement_events,
                'predict': self.predict_confinement_events,
            }

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

    def state_dict(self) -> dict:
        state = {}
        for item in self.state_items:
            state[item] = getattr(self, item)
        return state

    def load_state_dict(self, state: dict) -> None:
        print("Loading state_dict")
        for item in self.state_items:
            setattr(self, item, state[item])
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(f"Running Confinement_Datamodule.setup(stage={stage})")

        global_rank, world_size = self._get_rank_and_worldsize(self.trainer)
        self._global_rank = global_rank
        self._world_size  = world_size

        # Determine the dataset stage (train, validation, or test) based on the current stage
        if stage == 'fit':
            dataset_stages = ['train', 'validation']
        elif stage == 'test' or stage == 'predict':
            dataset_stages = [stage]
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Create DataLoaders for each dataset stage
        for dataset_stage in dataset_stages:
            # Determine the chunk of confinement indices for this GPU
            events = self.dataset_confinement_events[dataset_stage]
            times = [self.get_time_for_index(shot_event) for shot_event in events]  # Adapted for (shot, event) tuples
            if dataset_stage in ['train']:
                print(f"[setup] stage=train rank={global_rank}/{world_size} n_events={len(events)}")
                # Create balanced chunks
                chunks = self.create_balanced_chunks(events, times, world_size)
                # Determine the chunk for this GPU
                if len(chunks) == 0:
                    chunk_events = []
                else:
                    # SAFE indexing even if world_size/chunks mismatched
                    chunk_events = chunks[global_rank % len(chunks)]
            else:
                chunk_events = events

            if dataset_stage in ['train', 'validation', 'test']:
                dataset = self._load_and_preprocess_data(chunk_events, dataset_stage)
            elif dataset_stage in ['predict']:
                print(f"Preparing predict dataset for shots: {self.predict_shots}")
                # Load and preprocess the data for these events
                dataset = self._load_and_preprocess_predict_data(chunk_events)
                # Assign the predict dataset
                self.datasets["predict"] = dataset

            # Store the DataLoader for this GPU
            if dataset_stage == 'train':
                self._train_dataloader = torch.utils.data.DataLoader(
                                        dataset, 
                                        batch_size=self.batch_size,
                                        shuffle=True,             
                                        num_workers=self.num_workers,
                                        persistent_workers=(self.num_workers > 0),
                                        drop_last=True,
                                        )
                
    def _get_rank_and_worldsize(self, trainer):
        # try Lightning
        gr = getattr(trainer, "global_rank", None)
        ws = getattr(trainer, "world_size", None)

        # fall back to torch.distributed
        if (gr is None or ws is None) and torch.distributed.is_available() and torch.distributed.is_initialized():
            gr = torch.distributed.get_rank()
            ws = torch.distributed.get_world_size()

        # fall back to env (SLURM/torchrun)
        if gr is None:
            gr = int(os.getenv("RANK", "0"))
        if ws is None:
            ws = int(os.getenv("WORLD_SIZE", "1"))

        return int(gr), int(ws)
    
    def get_time_for_index(self, shot_event_tuple):
        shot, event = shot_event_tuple  # Unpack the tuple
        with h5py.File(self.data_file, 'r') as h5_file:
            event_key = f"{shot}/{event}"  # Updated to use shot/event structure
            time_count = h5_file[event_key]["signals"].shape[1]
        return time_count
    
    def create_balanced_chunks(self, indices, times, num_chunks):
        num_chunks = max(int(num_chunks or 1), 1)
        index_to_time = {idx: t for idx, t in zip(indices, times)}

        chunks = [[] for _ in range(num_chunks)]
        chunk_times = [0] * num_chunks

        for idx, t in sorted(index_to_time.items(), key=lambda kv: kv[1], reverse=True):
            j = min(range(num_chunks), key=lambda k: chunk_times[k])
            chunks[j].append(idx)
            chunk_times[j] += t

        # helpful debug on every rank
        print(f"[chunks] num_chunks={len(chunks)} sizes={[len(c) for c in chunks]} times={chunk_times}")
        return chunks
        
    def _load_and_preprocess_data(self, shot_event_indices, dataset_stage):
        t0 = time.time()
        print(f"Reading confinement events for dataset `{dataset_stage}`")

        # --------------------------
        # helpers
        # --------------------------

        from scipy.signal import decimate, resample_poly
        from fractions import Fraction
        from collections import Counter

        def _parse_label_filter(label_filter):
            if label_filter is None:
                return lambda v: True
            if isinstance(label_filter, (list, tuple, set, np.ndarray)):
                allowed = set(int(x) for x in label_filter)
                return lambda v: v in allowed
            if isinstance(label_filter, str):
                m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", label_filter)
                if not m:
                    raise ValueError(f"Unrecognized label_filter string: {label_filter!r}. Use 'a-b' or pass a list/set.")
                lo, hi = map(int, m.groups())
                return lambda v: lo <= v <= hi
            raise ValueError(f"Unrecognized label_filter type: {type(label_filter)}")

        def _downsample_1d(arr, orig_fs, target_fs, axis=0):
            if orig_fs == target_fs: return arr
            ratio = orig_fs / target_fs
            q = int(round(ratio))
            if abs(ratio - q) < 1e-6 and q >= 2:
                return decimate(arr, q, ftype='fir', axis=axis, zero_phase=True)
            frac = Fraction(target_fs / orig_fs).limit_denominator(64)
            return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

        def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
            x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
            new_len = x.shape[0]
            t = np.linspace(t_ms[0], t_ms[-1], new_len, dtype=np.float32)
            return x.astype(np.float32), t

        def _expected_down_len(N, orig_fs, target_fs):
            if target_fs is None or target_fs == orig_fs:
                return N
            ratio = orig_fs / target_fs
            q = int(round(ratio))
            if abs(ratio - q) < 1e-6 and q >= 2:
                # scipy.signal.decimate tends to behave like ceil(N/q) with zero_phase
                return int(np.ceil(N / q))
            else:
                from fractions import Fraction
                frac = Fraction(target_fs / orig_fs).limit_denominator(64)
                # resample_poly length behaves like ceil(N * up / down)
                return int(np.ceil(N * frac.numerator / frac.denominator))

        label_ok = _parse_label_filter(self.label_filter)

        confinement_data = []
        kept_indices = []            # list of (shot, event, label0, orig_len, new_len)
        time_counts_orig = []        # original lengths of kept events
        time_counts_ds = []          # downsampled lengths of kept events
        discards = Counter()         # {'short':..., 'missing_inboard':..., 'label':...}

        orig_fs = float(self.sampling_frequency_hz)         # typically 1e6
        tgt_fs  = float(self.target_sampling_hz) if getattr(self, 'target_sampling_hz', None) else None

        with h5py.File(self.data_file, 'r') as h5_file:
            if len(shot_event_indices) >= 5:
                print(f"  Initial shot/event indices: {shot_event_indices[:5]}")
            for i, (shot, event) in enumerate(shot_event_indices):
                event_key = f"{shot}/{event}"
                grp = h5_file[event_key]
                
                # Skip processing if inboard_order is missing or empty
                inboard_order = h5_file[shot].attrs.get("inboard_column_channel_order", None)
                if inboard_order is None or len(inboard_order) == 0:
                    print(f"Skipping event {event_key} due to missing or empty inboard_column_channel_order.")
                    discards['missing_inboard'] += 1
                    continue

                # Quick length check
                signal_length = grp["signals"].shape[1]
                if signal_length < self.signal_window_size:
                    discards['short'] += 1
                    continue

                # Peek a single label (events are uniquely labeled)
                # Safer check: we can still read only the first element; full array read is unnecessary here.
                lbl0 = int(grp["labels"][0])
                if not label_ok(lbl0):
                    discards['label'] += 1
                    continue

                # Compute post-downsample length for preallocation
                if tgt_fs is None or tgt_fs == orig_fs:
                    new_len = signal_length
                else:
                    new_len = _expected_down_len(signal_length, orig_fs, tgt_fs)
                    
                kept_indices.append((shot, event, lbl0, signal_length, new_len))
                time_counts_orig.append(signal_length)
                time_counts_ds.append(new_len)

        if not kept_indices:
            print("No events left after filtering; check your label_filter or window length.")
            # Keep behavior: return empties consistent with your pipeline
            return [], np.empty((0, self.n_rows, self.n_cols), dtype=np.float32)
            
        # Preallocate using *downsampled* lengths to avoid shape mismatch
        total_len = int(np.sum(time_counts_ds))
        packaged_signals = np.empty((total_len, self.n_rows, self.n_cols), dtype=np.float32)

        # Second pass: actually read/transform signals
        start_index = 0   
        with h5py.File(self.data_file, 'r') as h5_file:
            for i, (shot, event, lbl0, orig_len, new_len) in enumerate(kept_indices):
                if i % 100 == 0:
                    print(f"  Reading event {i:04d}/{len(shot_event_indices):04d} in shot {shot}")
                event_key = f"{shot}/{event}"
                event_data = h5_file[event_key]

                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = h5_file[shot].attrs["inboard_column_channel_order"]

                # Retrieve signals and reshape according to inboard_order
                signals = np.array(event_data["signals"][:, :], dtype=np.float32)
                signals = self.reshape_signals_6x8(signals, inboard_order)
                times   = np.array(event_data["time"][:], dtype=np.float32)        # (T,)

                start_col_index = (8-self.n_cols)
                signals = signals[:, :self.n_rows, start_col_index:]
                
                # signals = np.transpose(signals, (1, 0)).reshape(-1, self.n_rows, self.n_cols)
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    signals = self.apply_bandpass_filter(signals)

                # downsample to target rate
                if self.target_sampling_hz is not None:
                    orig_fs = float(self.sampling_frequency_hz)   # typically 1e6
                    tgt_fs  = float(self.target_sampling_hz)
                    signals, times = _downsample_stack(signals, times, orig_fs, tgt_fs)

                labels = np.full(signals.shape[0], lbl0, dtype=np.int64)

                labels, valid_t0 = self._get_valid_indices(labels)
                packaged_signals[start_index:start_index + signals.shape[0]] = signals
                start_index += signals.shape[0]
                confinement_data.append({
                    'labels': labels, 
                    'valid_t0': valid_t0,
                    'confinement_mode_key': event_key,
                    'shot': shot,
                    'time': event,
                })

        elapsed = time.time() - t0
        print(f"Kept {len(kept_indices)} events | "
            f"Discarded(short={discards['short']}, missing_inboard={discards['missing_inboard']}, label={discards['label']}) | "
            f"Total samples (downsampled)={total_len} | took {elapsed:.2f}s")

        # print(f"  Global min/max raw signal, ch 1-32: {np.amin(packaged_signals[:,:4,:]):.6f}, {np.amax(packaged_signals[:,:4,:]):.6f}")
        # print(f"  Global min/max raw signal, ch 33-64: {np.amin(packaged_signals[:,4:,:]):.6f}, {np.amax(packaged_signals[:,4:,:]):.6f}")

        packaged_labels = np.concatenate([confinement_mode['labels'] for confinement_mode in confinement_data], axis=0)
        # if self.one_hot_labels:
        #     encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(self.num_classes)], handle_unknown='ignore')
        #     packaged_labels = encoder.fit_transform(packaged_labels.reshape(-1, 1))

        # 1) Determine allowed labels from label_filter
        if self.label_filter is None:
            allowed = np.arange(self.num_classes, dtype=int)
        elif isinstance(self.label_filter, str):
            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", self.label_filter)
            lo, hi = map(int, m.groups())
            allowed = np.arange(lo, hi + 1, dtype=int)
        else:
            allowed = np.array(sorted(int(x) for x in self.label_filter), dtype=int)

        # 2) Remap original labels -> [0 .. K_eff-1] in allowed order
        old2new = {old: i for i, old in enumerate(allowed.tolist())}
        mask = np.isin(packaged_labels, allowed)
        assert mask.all(), f"Unexpected labels outside {allowed}: {np.unique(packaged_labels[~mask])}"
        remapped = np.vectorize(old2new.get)(packaged_labels).astype(int)

        # 3) If you want one-hot, build it with the *effective* K
        K_eff = len(allowed)
        if self.one_hot_labels:
            packaged_labels = np.eye(K_eff, dtype=np.float32)[remapped]
        else:
            packaged_labels = remapped  # integer targets for CE loss

        packaged_valid_t0 = np.concatenate([confinement_mode['valid_t0'] for confinement_mode in confinement_data], axis=0)
        # assert packaged_labels.size == packaged_valid_t0.size
        # assert packaged_labels.shape[0] == packaged_valid_t0.shape[0]

        # start indices for each confinement mode event in concatenated dataset
        packaged_window_start = []
        index = 0
        for confinement_mode in confinement_data:
            packaged_window_start.append(index)
            index += confinement_mode['labels'].size
        packaged_window_start = np.array(packaged_window_start, dtype=int)

        packaged_confinement_mode_key = np.array(
            [confinement_mode['confinement_mode_key'] for confinement_mode in confinement_data],
            dtype=str,
        )
        packaged_shot = np.array(
            [confinement_mode['shot'] for confinement_mode in confinement_data],
            dtype=int,
        )
        packaged_start_time = np.array(
            [confinement_mode['time'] for confinement_mode in confinement_data],
            dtype=int,
        )
        del confinement_data

        # valid t0 indices
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype=int)
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]
        assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices]))
        # assert np.all(np.isfinite(packaged_labels[packaged_valid_t0_indices + self.signal_window_size]))
        print("  Raw data stats")
        stats = self._get_statistics(
            sample_indices=packaged_valid_t0_indices,
            signals=packaged_signals,
        )

        # mask outlier signals
        if self.mask_sigma_outliers:
            if None in [self.mask_lb, self.mask_ub]:
                assert dataset_stage == 'train' or not self.train_confinement_events, f"Dataset_stage: {dataset_stage}"
                print(f"  Calculating mask upper/lower bounds from {dataset_stage} data")
                self.mask_lb = stats['mean'] - self.mask_sigma_outliers * stats['stdev']
                self.mask_ub = stats['mean'] + self.mask_sigma_outliers * stats['stdev']
                self.save_hyperparameters({
                    'mask_lb': self.mask_lb.item(),
                    'mask_ub': self.mask_ub.item(),
                })
            print(f"  Mask {self.mask_sigma_outliers:.2f} sigma outliers from signals")
            print(f"  Mask lower bound {self.mask_lb:.3f} upper bound {self.mask_ub:.3f}")
            mask = np.zeros(packaged_valid_t0_indices.size, dtype=bool)
            for i_t0_index, t0_index in enumerate(packaged_valid_t0_indices):
                signal_window = packaged_signals[t0_index: t0_index + self.signal_window_size, :, :]
                mask[i_t0_index] = (
                    np.max(signal_window) <= self.mask_ub and
                    np.min(signal_window) >= self.mask_lb
                )
            packaged_valid_t0_indices = packaged_valid_t0_indices[mask]
            print("  Masked data stats")
            stats = self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
        
        # standardize signals based on training data
        if None in [self.signal_mean, self.signal_stdev]:
            assert dataset_stage == 'train' or not self.train_confinement_events, f"Dataset_stage: {dataset_stage}"
            print(f"  Calculating signal mean and std from {dataset_stage} data")
            self.signal_mean = stats['mean']
            self.signal_stdev = stats['stdev']
            self.signal_exkurt = stats['exkurt']
            self.save_hyperparameters({
                'signal_mean': self.signal_mean.item(),
                'signal_stdev': self.signal_stdev.item(),
                'signal_exkurt': self.signal_exkurt.item(),
            })

        if dataset_stage in ['train'] and self.standardize_signals:
            print(f"  Standarizing signals with mean {self.signal_mean:.3f} and std {self.signal_stdev:.3f}")
            print(f"  Standardized signal stats")
            # packaged_signals = (packaged_signals - self.signal_mean) / self.signal_stdev
            for idx, signal in enumerate(packaged_signals):
                packaged_signals[idx] = (signal - self.signal_mean) / self.signal_stdev
            stats = self._get_statistics(
                sample_indices=packaged_valid_t0_indices,
                signals=packaged_signals,
            )
        self.max_abs_valid_signal = np.max(np.abs([stats['min'],stats['max']]))
                    
        if dataset_stage in ['train']:
            dataset = Confinement_TrainValTest_Dataset(
                signals=packaged_signals,
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                labels=packaged_labels,
                sample_indices=packaged_valid_t0_indices,
                window_start_indices=packaged_window_start,
                signal_window_size=self.signal_window_size,
                confinement_mode_keys=packaged_confinement_mode_key,
            )
            return dataset
        if dataset_stage in ['validation', 'test']:
            self.datasets[dataset_stage] = Confinement_TrainValTest_Dataset(
                    signals=packaged_signals,
                    n_rows=self.n_rows,
                    n_cols=self.n_cols,
                    labels=packaged_labels,
                    sample_indices=packaged_valid_t0_indices,
                    window_start_indices=packaged_window_start,
                    signal_window_size=self.signal_window_size,
                    confinement_mode_keys=packaged_confinement_mode_key,
                )
            return
        # if dataset_stage in ['predict']:
        #     del self._train_dataloader
        #     del self.datasets['validation']
            
        #     predict_datasets = []
        #     for i_confinement_mode, idx_start in enumerate(packaged_window_start):
        #         if self.max_predict_confinement_modes and i_confinement_mode == self.max_predict_confinement_modes:
        #             break
        #         if i_confinement_mode == packaged_window_start.size - 1:
        #             idx_stop = packaged_labels.size - 1
        #         else:
        #             idx_stop = packaged_window_start[i_confinement_mode+1]-1
        #         dataset = Confinement_Predict_Dataset(
        #             signals=packaged_signals[idx_start:idx_stop, ...],
        #             labels=packaged_labels[idx_start:idx_stop],
        #             signal_window_size=self.signal_window_size,
        #             shot=packaged_shot[i_confinement_mode],
        #             start_time=packaged_start_time[i_confinement_mode],
        #             confinement_mode_index=packaged_confinement_mode_key[i_confinement_mode],
        #         )
        #         predict_datasets.append(dataset)
        #     self.datasets['predict'] = predict_datasets
            # return predict_datasets
        # print(f"  Data stage `{dataset_stage}` elapsed time {(time.time()-t0)/60:.1f} min")
        gc.collect()
        torch.cuda.empty_cache()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)    

    def _load_and_preprocess_predict_data(self, shot_start_time_indices):
        """
        Build predict datasets for the multiclass confinement classifier.

        Returns
        -------
        List[Confinement_Predict_Dataset]
        Each dataset corresponds to one contiguous confinement-mode segment and yields:
        (signal_window[1,W,r,c], label[class_id], shot:int, start_time:int_ms)
        """
        import numpy as np, h5py, torch
        from scipy.signal import decimate, resample_poly
        from fractions import Fraction

        # ---------- helpers ----------
        def _parse_label_filter(label_filter):
            """Same semantics as training."""
            if label_filter is None:
                return lambda v: True
            if isinstance(label_filter, (list, tuple, set, np.ndarray)):
                allowed = set(int(x) for x in label_filter)
                return lambda v: v in allowed
            if isinstance(label_filter, str):
                m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", label_filter)
                if not m:
                    raise ValueError(f"Unrecognized label_filter string: {label_filter!r}. Use 'a-b' or pass a list/set.")
                lo, hi = map(int, m.groups())
                return lambda v: lo <= v <= hi
            raise ValueError(f"Unrecognized label_filter type: {type(label_filter)}")

        def _apply_label_filter_vectorized(labels_1d, label_filter, unknown_value=-1):
            """
            Map labels not in label_filter to unknown_value (e.g., -1 or np.nan).
            Works with int or float labels_1d; returns float if unknown_value is NaN.
            """
            a = np.asarray(labels_1d)
            if label_filter is None:
                # Preserve dtype unless unknown_value forces float
                if isinstance(unknown_value, float) and np.isnan(unknown_value):
                    return a.astype(np.float32)
                return a

            # Build a fast mask
            ok = _parse_label_filter(label_filter)
            # If a may contain NaN already, compare with care:
            if np.issubdtype(a.dtype, np.integer):
                mask = np.vectorize(lambda x: ok(int(x)))(a)
                out = a.astype(a.dtype, copy=True)
            else:
                # float dtype (allows NaN)
                # For NaNs, treat as not ok -> will become unknown_value
                mask = np.vectorize(lambda x: ok(int(x)) if not np.isnan(x) else False)(a)
                out = a.astype(np.float32, copy=True)

            # If unknown_value is NaN we must have float dtype
            if isinstance(unknown_value, float) and np.isnan(unknown_value):
                out = out.astype(np.float32, copy=False)
                out[~mask] = np.nan
            else:
                out[~mask] = unknown_value
            return out
        
        def _downsample_1d(arr, orig_fs, target_fs, axis=0):
            if orig_fs == target_fs: return arr
            ratio = orig_fs / target_fs
            q = int(round(ratio))
            if abs(ratio - q) < 1e-6 and q >= 2:
                return decimate(arr, q, ftype='fir', axis=axis, zero_phase=True)
            frac = Fraction(target_fs / orig_fs).limit_denominator(64)
            return resample_poly(arr, up=frac.numerator, down=frac.denominator, axis=axis)

        def _downsample_stack(sig_trc, t_ms, orig_fs, target_fs):
            """
            Downsample signals AND build a physically consistent timebase:
            t_ds[k] = t_ms[0] + k * (1000/target_fs).
            """
            x = _downsample_1d(sig_trc, orig_fs, target_fs, axis=0)
            new_len = x.shape[0]
            dt_ms = 1000.0 / float(target_fs)
            t0 = float(t_ms[0])
            t = (t0 + dt_ms * np.arange(new_len, dtype=np.float64)).astype(np.float32)
            return x.astype(np.float32), t, float(dt_ms)

        def _segment_runs(labels_1d: np.ndarray, unknown_sentinel=-1):
            """
            Return [(start_idx, stop_idx, class_id), ...] where stop_idx is exclusive.
            Treat NaN (if any) as unknown_sentinel.
            Works for int or float arrays.
            """
            a = np.asarray(labels_1d)
            if a.size == 0:
                return []

            if a.dtype.kind in "iu":   # integer dtype
                lab = a.astype(np.int64, copy=False)
            else:
                lab = a.copy()
                nan_mask = np.isnan(lab)
                if nan_mask.any():
                    lab[nan_mask] = unknown_sentinel
                lab = lab.astype(np.int64, copy=False)

            change = np.where(lab[1:] != lab[:-1])[0] + 1
            starts = np.r_[0, change]
            stops  = np.r_[change, lab.size]
            return [(int(s), int(e), int(lab[s])) for s, e in zip(starts, stops)]
        
        # ---------- config ----------
        W       = int(self.signal_window_size)
        hop     = int(getattr(self, "predict_window_stride", 1))          # <-- honored now
        orig_fs = float(self.sampling_frequency_hz)   # ~1e6
        tgt_fs  = float(self.target_sampling_hz)
        unknown_value = getattr(self, "predict_unknown_label_value", -1)        
        
        predict_datasets = []
        with h5py.File(self.data_file, 'r') as h5:
            # validate events
            valid_events = []
            for shot, event in shot_start_time_indices:
                skey = f"{shot}/{event}"
                if skey in h5 and 'signals' in h5[skey] and 'time' in h5[skey]:
                    valid_events.append((shot, event))
                else:
                    print(f"[confinement/predict] skip {skey}: missing signals/times")

            for i, (shot, event) in enumerate(valid_events):
                if i % 100 == 0:
                    print(f"[confinement/predict] {i:04d}/{len(valid_events):04d} shot={shot} event={event}")

                grp  = h5[f"{shot}/{event}"]
                shot_grp = h5[str(shot)]  # <--- use consistent string key

                # ---- load raw ----
                sig_xt = np.array(grp['signals'], dtype=np.float32)     # (X,T)
                t_ms  = np.array(grp['time'],   dtype=np.float64)     # (T,)
                if sig_xt.shape[1] < W:   # not enough samples for one window
                    continue

                # ---- reshape to (T, R_full, C_full) and select (r,c) ----
                # Retrieve the inboard_column_channel_order for this shot
                inboard_order = shot_grp.attrs["inboard_column_channel_order"]
                signals = self.reshape_signals_6x8(sig_xt, inboard_order)
                start_col_index = (8-self.n_cols)
                sig_trc = signals[:, :self.n_rows, start_col_index:]

                # optional bandpass for classifier
                if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None:
                    if i % 100 == 0:
                        print(f"  applying {self.lower_cutoff_frequency_hz} - {self.upper_cutoff_frequency_hz} bandpass filter ")
                    sig_trc = self.apply_bandpass_filter(sig_trc)

                # ---- downsample ----
                if self.target_sampling_hz is not None:
                    sig_trc, t_ms, dt_ms = _downsample_stack(sig_trc, t_ms, orig_fs, tgt_fs)   # (T_ds,r,c), (T_ds,), float

                # ---- standardize using train stats (important!) ----
                if getattr(self, "standardize_signals", False):
                    assert self.signal_mean is not None and self.signal_stdev is not None, \
                        "Predict requires training mean/stdev to standardize inputs."
                    sig_trc = (sig_trc - self.signal_mean) / max(self.signal_stdev, 1e-12)

                # ---- labels (align to t_ds) ----
                labels_1d = None
                time_key = 'time' if 'time' in grp else ('times' if 'times' in grp else None)
                if labels_1d is None:
                    # event datasets: labels + time (or times)
                    time_key = 'time' if 'time' in grp else ('times' if 'times' in grp else None)
                    if 'labels' in grp and time_key is not None:
                        lab_t = np.array(grp[time_key], dtype=np.float32)    # original event timebase
                        lab_v = np.array(grp['labels'], dtype=np.int64)      # class ids 0..C-1
                        # align to t_ds (downsampled)
                        idx  = np.searchsorted(lab_t, t_ms, side='left')
                        idx0 = np.clip(idx - 1, 0, lab_t.size - 1)
                        idx1 = np.clip(idx,       0, lab_t.size - 1)
                        d0   = np.abs(t_ms - lab_t[idx0])
                        d1   = np.abs(t_ms - lab_t[idx1])
                        use0 = d0 <= d1
                        nn   = np.where(use0, idx0, idx1)
                        labels_1d = lab_v[nn].astype(np.int64)
                        self._ensure_label_maps()  # builds allowed_labels, label_to_local, etc.
                        labels_local = self._remap_to_local(labels_1d, unknown_value=unknown_value)

                    # labels_1d are original IDs (0..C-1). Remap to local [0..K_eff-1]; unknown -> -1
                    unknown_value = getattr(self, "predict_unknown_label_value", -1)
                    labels_local = self._remap_to_local(labels_1d, unknown_value=unknown_value)

                    # segment on local labels
                    spans = _segment_runs(labels_local, unknown_sentinel=unknown_value)

                    if not spans:
                        continue

                # ---- per-span sliding windows, build dataset objects ----
                for s_idx, e_idx, class_id in spans:
                    # candidate window ends (inclusive) inside [s_idx, e_idx)
                    # (we use ends to be consistent with your label-at-window-end convention)
                    end_idxs = np.arange(max(s_idx, W-1), e_idx, hop, dtype=int)
                    if end_idxs.size == 0:
                        continue
                    # convert to window STARTS relative to the span:
                    t0_candidates = (end_idxs - (W - 1)).astype(int)     # starts in [s_idx - (W-1), ...], all >= s_idx by construction
                    # restrict signals/labels to the span:
                    sig_span = sig_trc[s_idx:e_idx]                       # (T_seg,r,c)
                    lab_span = labels_local[s_idx:e_idx]  # keep local IDs (or -1) per time step
                    # time for the FIRST window end in this span (absolute ms)
                    first_end_abs_ms = float(t_ms[s_idx + (W - 1)])
                    # sample indices relative to the SPAN (dataset-local):
                    local_starts = t0_candidates - s_idx                 # 0 .. (T_seg - W)
                    assert np.all(local_starts >= 0)

                    ds = Confinement_Predict_Dataset(
                                    signals=sig_span,                     # (T_seg, r, c)
                                    n_rows=self.n_rows,
                                    n_cols=self.n_cols,
                                    labels=lab_span,                      # (T_seg,)
                                    signal_window_size=W,
                                    shot=int(shot),
                                    start_time=int(round(first_end_abs_ms)),   # will be used as FIRST window-end time
                                    confinement_mode_index=int(class_id),
                                    sample_indices=torch.from_numpy(local_starts.astype(np.int64)),
                                    dt_ms=float(dt_ms)                    # needed to advance time per window
                                )
                    predict_datasets.append(ds)

        if len(predict_datasets) == 0:
            print("[confinement/predict] no samples prepared for classification.")
            return ConcatDataset([])  # or a dummy empty dataset
        else:
            print(f"[confinement/predict] built {len(predict_datasets)} segment datasets.")
            return ConcatDataset(predict_datasets)

    def reshape_signals_6x8(self, signals, inboard_order):
        # Assumptions:
        # - `inboard_order` contains valid indices for the starting positions of each row.
        # - `signals` is expected to be of shape (num_channels, num_samples), where num_channels >= max(inboard_order) + 7.

        # truncate the inboard_order array to first 6 rows
        inboard_order = inboard_order[:6]

        # Initialize the reshaped signals with zeros or np.nan if there's a chance of not filling some cells
        reshaped_signals = np.zeros((signals.shape[1], 6, 8), dtype=np.float32)  # Using zeros as default values

        # Reshape signals according to the truncated inboard_order
        for row, start_idx in enumerate(inboard_order):
            for col in range(8):
                channel_idx = start_idx + col - 1  # Adjusting for 0-indexing if inboard_order is 1-indexed

                # Ensure the calculated index is within the bounds of the signals array
                if 0 <= channel_idx < signals.shape[0]:
                    reshaped_signals[:, row, col] = signals[channel_idx, :]
                else:
                    print(f"Warning: Channel index {channel_idx} out of bounds for row {row}, col {col}.")

        return reshaped_signals


    def apply_bandpass_filter(self, signals):
            """
            Applies a bandpass filter to the given signals if the cutoff frequencies are specified.
            Otherwise, returns the original signals.

            Args:
                packaged_signals: The signals to be filtered.

            Returns:
                Filtered signals or the original signals.
            """
            required_length = 3 * self.filter_taps  # Set to 3 times the number of filter taps

            # Check if the cutoff frequencies are specified
            if self.lower_cutoff_frequency_hz is not None and self.upper_cutoff_frequency_hz is not None and signals.shape[0] > required_length:
                # Design the bandpass filter
                bandpass_filter = firwin(
                    self.filter_taps,
                    [self.lower_cutoff_frequency_hz, self.upper_cutoff_frequency_hz],
                    pass_zero=False,
                    fs=self.sampling_frequency_hz
                )

                # Apply the filter
                filtered_signals = filtfilt(bandpass_filter, 1, signals, axis=0)
                return filtered_signals
            else:
                # print("BANDPASS FILTER NOT APPLIED")
                return signals

    def _get_confinement_events_and_split(self):
        print(f"Data file: {self.data_file}")
        shots = self._load_and_label_shots()
        self._apply_metadata_filters(shots)
        self._apply_forced_shots(shots)
        filtered_shots = self._filter_shots_by_class(shots)
        self._shuffle_and_limit_shots(filtered_shots)
        self._split_datasets(filtered_shots)
        self.calculate_mode_times_and_shots(self.data_file, list(filtered_shots.keys()))

        # 3) NEW: build predict events purely from user-specified predict_shots
        self._set_predict_confinement_events_from_shots(shots)

    def _load_and_label_shots(self):
        if self.bad_shots is None:
            self.bad_shots = []

        shots = {}
        with h5py.File(self.data_file, "r") as data_file:
            for shot in data_file.keys():
                if shot in self.bad_shots:
                    print(f"Skipping bad shot: {shot}")
                    continue

                shot_labels = self._collect_labels(data_file, shot)

                # Avoid ambiguous truth checks on arrays:
                if not shot_labels:   # empty list
                    continue

                label_presence = self._label_presence(shot_labels)

                # Build (shot, event) only for groups that actually contain 'labels'
                shot_events = []
                for event_name, obj in data_file[shot].items():
                    if isinstance(obj, h5py.Group) and 'labels' in obj.keys():
                        shot_events.append((shot, event_name))

                metadata = self._extract_metadata(data_file[shot].attrs)
                shots[shot] = (shot_events, label_presence, metadata)

        return shots

    def _apply_metadata_filters(self, shots):
        r_avg_exclusions = z_avg_exclusions = delz_avg_exclusions = 0
        for shot in list(shots):
            metadata = shots[shot][2]
            if not self._metadata_within_bounds(metadata):
                shots.pop(shot)
                if metadata['r_avg'] is None or not self._check_bounds(metadata['r_avg'], self.metadata_bounds['r_avg']):
                    r_avg_exclusions += 1
                if metadata['z_avg'] is None or not self._check_bounds(metadata['z_avg'], self.metadata_bounds['z_avg']):
                    z_avg_exclusions += 1
                if metadata['delz_avg'] is None or not self._check_bounds(metadata['delz_avg'], self.metadata_bounds['delz_avg']):
                    delz_avg_exclusions += 1
        print(f"Number of r_avg exclusions: {r_avg_exclusions}")
        print(f"Number of z_avg exclusions: {z_avg_exclusions}")
        print(f"Number of delz_avg exclusions: {delz_avg_exclusions}")

    def _metadata_within_bounds(self, metadata):
        return all(self._check_bounds(metadata[key], self.metadata_bounds[key]) for key in ['r_avg', 'z_avg', 'delz_avg'] if key in self.metadata_bounds)

    def _check_bounds(self, value, bounds):
        return bounds[0] <= value <= bounds[1] if bounds else True

    def _extract_metadata(self, attrs):
        return {
            'r_avg': attrs.get('r_avg'),
            'z_avg': attrs.get('z_avg'),
            'delz_avg': attrs.get('delz_avg')
        }
    
    def _apply_forced_shots(self, shots):
        test_shot_data = {}
        validation_shot_data = {}

        # Handling forced test shots
        if self.force_test_shots:
            for shot_number in self.force_test_shots:
                if shot_number in shots:
                    test_shot_data[shot_number] = shots.pop(shot_number)
                else:
                    print(f"Warning: Forced test shot number {shot_number} not found in dataset.")

        # Handling forced validation shots
        if self.force_validation_shots:
            for shot_number in self.force_validation_shots:
                if shot_number in shots:
                    validation_shot_data[shot_number] = shots.pop(shot_number)
                else:
                    print(f"Warning: Forced validation shot number {shot_number} not found in dataset.")

        # These dictionaries can be used to ensure that the specified shots are included in their respective datasets
        self.forced_test_shots_data = test_shot_data
        self.forced_validation_shots_data = validation_shot_data

         # Return shots including forced shots for further processing
        return {**shots, **test_shot_data, **validation_shot_data}
    
    def _compute_allowed_from_filter(self, label_filter, num_classes):
        if label_filter is None:
            return np.arange(num_classes, dtype=int)
        if isinstance(label_filter, str):
            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", label_filter)
            if not m:
                raise ValueError(f"Bad label_filter: {label_filter!r}")
            lo, hi = map(int, m.groups())
            return np.arange(lo, hi + 1, dtype=int)
        return np.array(sorted(int(x) for x in label_filter), dtype=int)

    def _ensure_label_maps(self):
        # Build once and reuse everywhere (train/val/test/predict)
        if getattr(self, "allowed_labels", None) is None:
            allowed = self._compute_allowed_from_filter(self.label_filter, self.num_classes)
            self.allowed_labels = allowed.astype(int)
            self.effective_num_classes = int(len(allowed))
            self.label_to_local = {old: i for i, old in enumerate(self.allowed_labels.tolist())}
            self.local_to_label = self.allowed_labels.copy()  # np.array of original IDs in local order
            # Optional: persist
            self.save_hyperparameters({
                "allowed_labels": self.allowed_labels.tolist(),
                "effective_num_classes": self.effective_num_classes,
            })

    def _remap_to_local(self, labels_1d, unknown_value=-1):
        """Map original class IDs -> local [0..K_eff-1]; others -> unknown_value."""
        a = np.asarray(labels_1d, dtype=np.int64)
        out = np.full_like(a, fill_value=unknown_value)
        for old, new in self.label_to_local.items():
            out[a == old] = new
        return out

    def _collect_labels(self, data_file, shot):
        shot_labels = []
        shot_grp = data_file[shot]

        for event_name, obj in shot_grp.items():
            # Only check membership on groups
            if isinstance(obj, h5py.Group) and 'labels' in obj.keys():
                ds = obj['labels']
                # string vs numeric handling
                if h5py.check_string_dtype(ds.dtype) is not None:
                    arr = ds.asstr()[()]                 # read as str
                else:
                    arr = np.asarray(ds[()], dtype=None) # read as np array

                # Flatten to 1D then extend the Python list
                shot_labels.extend(np.ravel(arr).tolist())

        return shot_labels

    def _label_presence(self, labels):
        return tuple(class_id in labels for class_id in range(self.num_classes))  

    def _filter_shots_by_class(self, shots):
        shots_by_class = {}
        for shot, (events, labels, metadata) in shots.items():
            if labels not in shots_by_class:
                shots_by_class[labels] = []
            shots_by_class[labels].append(shot)
        if self.max_shots_per_class is not None:
            for labels, shot_list in shots_by_class.items():
                if len(shot_list) > self.max_shots_per_class:
                    shots_by_class[labels] = np.random.choice(shot_list, self.max_shots_per_class, replace=False).tolist()
        return {shot: shots[shot] for label_shots in shots_by_class.values() for shot in label_shots}

    def _shuffle_and_limit_shots(self, filtered_shots):
        shot_numbers = np.array(list(filtered_shots.keys()))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(shot_numbers)
        if self.max_shots:
            shot_numbers = shot_numbers[:self.max_shots]
        self.all_confinement_events = np.concatenate([filtered_shots[shot][0] for shot in shot_numbers])

    def _split_datasets(self, filtered_shots):
        if not self.test_only:
            shot_numbers = np.array(list(filtered_shots.keys()))
            # Map labels here
            # labels = [self.map_labels(filtered_shots[shot][1]) for shot in shot_numbers]
            labels = [filtered_shots[shot][1] for shot in shot_numbers]
            # Try to stratify, revert to random split if stratification is not possible
            try:
                train_indices, test_val_indices = train_test_split(shot_numbers, labels, test_size=self.fraction_test + self.fraction_validation, stratify=labels, random_state=self.seed)
            except ValueError:
                print("Stratified split failed; reverting to random split for train/test+validation sets.")
                train_indices, test_val_indices = train_test_split(shot_numbers, test_size=self.fraction_test + self.fraction_validation, random_state=self.seed)
            try:
                test_indices, val_indices = train_test_split(
                    test_val_indices,
                    [filtered_shots[shot][1] for shot in test_val_indices],
                    test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
                    # stratify=[self.map_labels(filtered_shots[shot][1]) for shot in test_val_indices],
                    stratify=[filtered_shots[shot][1] for shot in test_val_indices],
                    random_state=self.seed
                )
            except ValueError:
                print("Stratified split failed; reverting to random split for test/validation sets.")
                test_indices, val_indices = train_test_split(
                    test_val_indices,
                    test_size=self.fraction_validation/(self.fraction_test + self.fraction_validation),
                    random_state=self.seed  # No stratification here
                )

            # Ensure forced shots are included back in filtered_shots if needed
            filtered_shots.update(self.forced_test_shots_data)
            filtered_shots.update(self.forced_validation_shots_data)

            # Include forced test and validation shots
            if hasattr(self, 'forced_test_shots_data'):
                forced_test_indices = np.array(list(self.forced_test_shots_data.keys()))
                test_indices = np.concatenate((test_indices, forced_test_indices))
            
            if hasattr(self, 'forced_validation_shots_data'):
                forced_val_indices = np.array(list(self.forced_validation_shots_data.keys()))
                val_indices = np.concatenate((val_indices, forced_val_indices))

            # Assign events to datasets
            self.train_confinement_events = [event for shot in train_indices for event in filtered_shots[shot][0]]
            self.validation_confinement_events = [event for shot in val_indices for event in filtered_shots[shot][0]]
            self.test_confinement_events = [event for shot in test_indices for event in filtered_shots[shot][0]]

            print(f"Train set size: {len(self.train_confinement_events)} events")
            print(f"Validation set size: {len(self.validation_confinement_events)} events")
            print(f"Test set size: {len(self.test_confinement_events)} events")

            print(f"Train shot numbers: {train_indices}")
            print(f"Validation shot numbers: {val_indices}")
            print(f"Test shot numbers: {test_indices}")

        else:
            shot_numbers = np.array(list(filtered_shots.keys()))
            self.test_confinement_events = [event for shot in shot_numbers for event in filtered_shots[shot][0]]


    def map_labels(self, label):
        # Define less common labels
        uncommon_labels_0 = [
            (True, True, False, False),
            (False, True, True, False),
            (True, False, True, True),
            (False, False, True, True), 
            # (True, True, True, False), # Add other similar rare labels if needed
        ]
        uncommon_labels_1 = [
            (True, False, True, False),  # Add other similar rare labels if needed
        ]
        # Map less common labels to a new, single label
        if label in uncommon_labels_0:
            return (True, True, True, True)  # Example: re-map to this label
        elif label in uncommon_labels_1:
            return (True, False, False, False)
        return label
    
    def calculate_mode_times_and_shots(self, file_path, valid_shot_keys):
        mode_times = defaultdict(float)  # time accumulated per mode
        mode_shots = defaultdict(set)    # unique shot IDs per mode

        # helper to find a time key in a group
        def _find_time_key(g):
            for k in ("time", "times", "time_centers", "label_times"):
                if k in g.keys():
                    return k
            return None

        with h5py.File(file_path, "r") as f:
            for shot_key in valid_shot_keys:
                if shot_key not in f:
                    continue
                shot_grp = f[shot_key]

                for nested_name, obj in shot_grp.items():
                    # only do membership checks on groups
                    if not isinstance(obj, h5py.Group):
                        continue

                    if "labels" not in obj.keys():
                        continue

                    tkey = _find_time_key(obj)
                    if tkey is None:
                        continue

                    labels = np.asarray(obj["labels"][()], dtype=float)  # allow NaN
                    time   = np.asarray(obj[tkey][()], dtype=float)

                    # need at least two time points to form a diff
                    n = min(labels.shape[0], time.shape[0])
                    if n < 2:
                        continue

                    labels = labels[:n]
                    time   = time[:n]

                    dt = np.diff(time)               # length n-1
                    valid = ~np.isnan(labels[:-1])   # only intervals whose starting label is valid
                    if not np.any(valid):
                        continue

                    # iterate only over valid indices
                    for i in np.where(valid)[0]:
                        lab = int(labels[i])
                        mode_times[lab] += float(dt[i])
                        mode_shots[lab].add(shot_key)

        # Convert from ms → s (keep as in your original)
        mode_times_seconds = {k: v / 1e3 for k, v in mode_times.items()}
        mode_shot_counts   = {k: len(v)   for k, v in mode_shots.items()}

        print("Total Time Spent in Each Mode (seconds):", mode_times_seconds)
        print("Number of Unique Shots for Each Mode:", mode_shot_counts)

    def _set_predict_confinement_events_from_shots(self, shots_all: dict):
        """
        Build predict events using ALL events from user-provided self.predict_shots,
        regardless of train/val/test filtering.
        """
        predict_shots = getattr(self, "predict_shots", None) or []
        if not predict_shots:
            self.predict_confinement_events = []
            print("Predict set size: 0 events (no predict_shots provided)")
            return

        # normalize shot ids to strings to match HDF5 keys
        predict_shots_str = [str(s) for s in predict_shots]
        available = set(shots_all.keys())
        missing = [s for s in predict_shots_str if s not in available]
        if missing:
            print(f"Warning: The following predict shots are missing from the data file: {missing}")

        chosen = [s for s in predict_shots_str if s in available]
        predict_events = [evt for s in chosen for evt in shots_all[s][0]]  # shots_all[shot][0] = list[(shot, event)]

        self.predict_confinement_events = predict_events
        print(f"Predict set size: {len(self.predict_confinement_events)} events from shots {chosen}")

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Determine valid t0 indices (start of signal windows) for real-time inference
        valid_t0 = np.zeros(labels.size, dtype=int)
        first_valid_signal_window_start_index = self.signal_window_size - 1
        valid_t0[first_valid_signal_window_start_index:] = 1

        return labels, valid_t0
    
    def _get_statistics(
            self, 
            sample_indices: np.ndarray, 
            signals: np.ndarray,
    ) -> dict:
        signal_min = np.array(np.inf)
        signal_max = np.array(-np.inf)
        n_bins = 200
        cummulative_hist = np.zeros(n_bins, dtype=int)
        stat_samples = int(100e3)
        stat_interval = np.max([1, sample_indices.size//stat_samples])
        n_samples = sample_indices.size // stat_interval
        for i in sample_indices[::stat_interval]:
            signal_window = signals[i: i + self.signal_window_size, :, :]
            signal_min = np.min([signal_min, signal_window.min()])
            signal_max = np.max([signal_max, signal_window.max()])
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
        print(f"    Stats: count {sample_indices.size:,} min {signal_min:.3f} max {signal_max:.3f} mean {mean:.3f} stdev {stdev:.3f} exkurt {exkurt:.3f} n_samples {n_samples:,}")
        return {
            'count': sample_indices.size,
            'min': signal_min,
            'max': signal_max,
            'mean': mean,
            'stdev': stdev,
            'exkurt': exkurt,
        }

    def train_dataloader(self):
        return self._train_dataloader
    
    def val_dataloader(self):
        valid_sampler = torch.utils.data.DistributedSampler(
            self.datasets['validation'],
            shuffle=False,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=valid_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        ) 
                
    def test_dataloader(self):
        test_sampler = torch.utils.data.DistributedSampler(
            self.datasets['test'],
            shuffle=False,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            sampler=test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        ) 
    
    def predict_dataloader(self):
        predict_sampler = torch.utils.data.DistributedSampler(
            self.datasets['predict'],
            shuffle=False,
            drop_last=True,
        )
        return torch.utils.data.DataLoader(
            dataset=self.datasets['predict'],
            sampler=predict_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if self.num_workers > 0 else 0,  # Adjust based on the environment
            pin_memory=True,  # Enable pin_memory for faster data transfers to GPU
            persistent_workers=(self.num_workers > 0),  # Keep workers alive if num_workers > 0
        ) 
