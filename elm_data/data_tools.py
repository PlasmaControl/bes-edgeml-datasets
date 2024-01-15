import time
from datetime import datetime
from typing import Iterable
import os
import sys
import csv
import threading
import concurrent.futures
import contextlib
from pathlib import Path
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
import MDSplus
import h5py


def print_hdf5_contents(
        hdf5_file: Path|str,
        print_attributes: bool = True,
        print_datasets: bool = True,
        max_groups: int = 4,
):

    def _print_attributes(obj: h5py.Group|h5py.Dataset):
        more_indent = '  ' if isinstance(obj, h5py.Dataset) else ''
        for key in obj.attrs:
            item = obj.attrs[key]
            if isinstance(item, np.ndarray):
                print(more_indent + f'  Attribute {key}: shape {item.shape} dtype {item.dtype}')
            elif isinstance(item, str):
                print(more_indent + f'  Attribute {key}: {item}')
            elif isinstance(item, Iterable):
                print(more_indent + f'  Attribute {key}: len {len(item)}')
            else:
                print(more_indent + f'  Attribute {key}: value {item} type {type(item)}')

    def _recursively_print_content(group: h5py.Group):
        n_subgroups = 0
        n_datasets = 0
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                n_subgroups += 1
            elif isinstance(item, h5py.Dataset):
                n_datasets += 1
            else:
                raise ValueError
        print(f'Group {group.name}: {n_datasets} datasets, {n_subgroups} subgroups, and {len(group.attrs)} attributes')
        if print_attributes: _print_attributes(group)
        n_groups = 0
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                n_groups += 1
                if max_groups and n_groups <= max_groups:
                    _recursively_print_content(item)
            elif isinstance(item, h5py.Dataset):
                if print_datasets: print(f'  Dataset {key}:', item.shape, item.dtype, item.nbytes)
                if print_attributes: _print_attributes(item)
            else:
                raise ValueError

    hdf5_file = Path(hdf5_file)
    if not hdf5_file.exists():
        print(f'Data file does not exist: {hdf5_file}')
        return
    print(f'Contents of {hdf5_file}')
    with h5py.File(hdf5_file, 'r') as root:
        _recursively_print_content(root)


def make_mdsplus_connection() -> MDSplus.Connection:
    tries = 0
    success = False
    while success is False:
        try:
            connection = MDSplus.Connection('atlas.gat.com')
            success = True
        except:
            tries += 1
            time.sleep(2)
            if tries > 5:
                raise
    return connection


@dataclasses.dataclass
class Shot_Data:
    shot: int = 162303
    bes_channels: Iterable[int]|str = None  # list with 1-64 or 'all'
    max_bes_sample_rate: float = None  # Hz
    with_limited_signals: bool = False
    skip_metadata: bool = False
    only_8x8: bool = False
    only_standard_8x8: bool = False
    only_pos_ip: bool = False
    only_neg_bt: bool = False
    min_pinj_15l: float = 500e3  # W
    min_sustained_15l: float = 200  # ms
    mdsplus_connection: MDSplus.Connection = None
    quiet: bool = False

    def __post_init__(self):
        if self.mdsplus_connection is None:
            self.mdsplus_connection = make_mdsplus_connection()
        if self.bes_channels is None:
            self.bes_channels = []
        elif isinstance(self.bes_channels, str) and self.bes_channels == 'all':
            self.bes_channels = np.arange(1, 65, dtype=int)
        self.bes_channels = np.array(self.bes_channels, dtype=int)
        self.bes_time = None
        self.bes_signals = None
        self.get_bes_data()
        if self.skip_metadata is False:
            self.get_metadata()
        if self.with_limited_signals:
            self.get_limited_siganls()

    def get_bes_data(self):
        t1 = time.time()
        # get BES time array
        self.bes_time = np.array(
            self.mdsplus_connection.get(f'dim_of(ptdata("besfu01", {self.shot}))'),
            dtype=np.float32,
        )
        assert self.bes_time.size > 0, f'Shot {self.shot} has bad BES time data'
        # get BES signals
        if self.bes_channels.size > 0:
            assert np.all(self.bes_channels > 0), f'Shot {self.shot}: BES does not have channel `0`'
            self.bes_signals = np.empty([self.bes_channels.size, self.bes_time.size], dtype=np.float32)
            tdi_vars = []
            tdi_assignments = []
            for channel in self.bes_channels:
                var = f'_n{channel:02d}_{self.shot}'
                tdi_vars.append(var)
                tmp = f'{var} = ptdata("besfu{channel:02d}", {self.shot})'
                tdi_assignments.append(tmp)
            self.mdsplus_connection.get(', '.join(tdi_assignments))
            for i, tdi_var in enumerate(tdi_vars):
                self.bes_signals[i, :] = np.array(self.mdsplus_connection.get(tdi_var), dtype=np.float32)
            if self.max_bes_sample_rate:
                sample_rate = 1 / np.mean(np.diff(self.bes_time[:1000]/1e3))  # Hz
                downsample_factor = int(np.rint(sample_rate / self.max_bes_sample_rate))
                if downsample_factor >= 2:
                    self.bes_signals = self.bes_signals[:, ::downsample_factor]
                    self.bes_time = self.bes_time[::downsample_factor]
        if not self.quiet: print(f'{self.shot}: BES data time = {time.time() - t1:.2f} s')

    def get_metadata(self):
        t1 = time.time()
        # get BES configuration
        self.is_8x8 = self.is_standard_8x8 = False
        self.only_8x8 = self.only_8x8 or self.only_standard_8x8
        self.mdsplus_connection.openTree('bes', self.shot)
        self.r_position = np.array(self.mdsplus_connection.get(r'\bes_r'), dtype=np.float32)
        self.z_position = -np.array(self.mdsplus_connection.get(r'\bes_z'), dtype=np.float32)
        assert self.r_position.size == 64 and self.z_position.size == 64, f'Shot {self.shot} has bad BES position data'
        self.mdsplus_connection.closeTree('bes', self.shot)
        for i in np.arange(8):
            # del-r for column i
            rdiff = np.diff(self.r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.12)
            # del-z for row i
            zdiff = np.diff(self.z_position[i * 8 + np.arange(8)])
            row_test = np.all(np.abs(zdiff) <= 0.12)
            self.is_8x8 = col_test and row_test
            if self.is_8x8 is False:
                break
        if self.only_8x8:
            assert self.is_8x8, f'Shot {self.shot} is not 8x8'
        self.delz_avg = self.r_avg = self.z_avg = self.inboard_column_channel_order = None
        if self.is_8x8:
            self.delz_avg = np.abs(np.diff(self.z_position[np.arange(8)*8]).mean()).round(2)
            self.r_avg = np.mean(self.r_position).round(1)
            self.z_avg = np.mean(self.z_position).round(1)
            z_first_column = self.z_position[np.arange(8) * 8]
            self.inboard_column_channel_order = np.flip(z_first_column.argsort()) * 8
            self.is_standard_8x8 = np.array_equal(
                self.inboard_column_channel_order,
                np.arange(8, dtype=int) * 8,
            )
        if self.only_standard_8x8:
            assert self.is_standard_8x8, f'Shot {self.shot} is not standard 8x8'
        # get ip, beams, etc.
        for node_name in ['ip', 'bt']:
            result = self._get_signal(node_name, max_sample_rate=5e3)
            signal:np.ndarray = result['data']
            setattr(self, node_name, signal)
            setattr(self, f'{node_name}_time', result['time'])
            extremum = signal.max() if np.abs(signal.max()) > np.abs(signal.min()) else signal.min()
            setattr(self, f'{node_name}_extremum', extremum)
            setattr(self, f'{node_name}_pos_phi', True if extremum>0 else False)
        if self.only_pos_ip:
            assert self.ip_pos_phi is True, f'Shot {self.shot} has negative Ip (pos is normal)'
        if self.only_neg_bt:
            assert self.bt_pos_phi is False, f'Shot {self.shot} has positive Bt (neg is normal)'
        # mask for early Ip termination
        ip_mask = np.flatnonzero(self.ip >= 400e3)
        ip_stop_time = self.ip_time[ip_mask[-1]]
        self.bes_start_time = self.bes_time[0]
        self.bes_stop_time = self.bes_time[-1]
        if ip_stop_time - 100 < self.bes_stop_time:
            bes_mask = self.bes_time <= ip_stop_time-100
            self.bes_time = self.bes_time[bes_mask]
            if self.bes_signals is not None:
                self.bes_signals = self.bes_signals[:, bes_mask]
            self.bes_start_time = self.bes_time[0]
            self.bes_stop_time = self.bes_time[-1]
            for node_name in ['ip', 'bt']:
                signal = getattr(self, node_name)
                signal_time = getattr(self, f'{node_name}_time')
                mask = signal_time <= self.bes_stop_time
                setattr(self, node_name, signal[mask])
                setattr(self, f'{node_name}_time', signal_time[mask])
        # get NB
        for node_name in ['pinj', 'pinj_15l', 'pinj_15r']:
            result = self._get_signal(node_name, tree='nb', max_sample_rate=5e3)
            setattr(self, node_name, result['data'])
            setattr(self, f'{node_name}_max', result['data'].max())
            if node_name == 'pinj':
                setattr(self, f'{node_name}_time', result['time'])
                self.mdsplus_connection.openTree('nb', self.shot)
                date = self.mdsplus_connection.get(f'getnci(\\pinj, "time_inserted")')
                self.date = str(date.date)
                self.mdsplus_connection.closeTree('nb', self.shot)
        # check for minimum PINJ_15L power
        if self.min_pinj_15l:
            assert self.pinj_15l_max > self.min_pinj_15l, f'Shot {self.shot} has low PINJ_15l'
        # check for minimum sustained PINJ_15L
        if self.min_sustained_15l:
            mask = self.pinj_15l >= 500e3
            change = np.diff(np.array(mask, dtype=int))
            i_on = np.flatnonzero(change == 1)
            i_off = np.flatnonzero(change == -1)
            if i_off[0] > i_on[0] and i_on.size == i_off.size:
                valid = False
                for i1, i2 in zip(i_on, i_off):
                    delt = self.pinj_time[i2] - self.pinj_time[i1]
                    if delt >= self.min_sustained_15l:
                        valid = True
                        break
                assert valid, f'Shot {self.shot} does not have sustained PINJ_15l'
        if not self.quiet: print(f'{self.shot}: Metadata time = {time.time() - t1:.2f} s')

    def get_limited_siganls(self):
        t1 = time.time()
        node_names = ['FS03', 'FS04', 'FS05']
        for i_node, node_name in enumerate(node_names):
            signal_dict = self._get_signal(
                node_name=node_name,
                tree='spectroscopy',
                max_sample_rate=50e3,
            )
            setattr(self, node_name, signal_dict['data'])
            if i_node==0:
                setattr(self, 'FS_time', signal_dict['time'])
        node_names = ['denv3f']
        for i_node, node_name in enumerate(node_names):
            signal_dict = self._get_signal(
                node_name=node_name,
                tree='electrons',
                max_sample_rate=200e3,
            )
            setattr(self, node_name, signal_dict['data'])
            if i_node==0:
                setattr(self, f'{node_name}_time', signal_dict['time'])
        if not self.quiet: print(f'{self.shot}: Non-BES signal time = {time.time() - t1:.2f} s')

    def _get_signal(
            self,
            node_name: str,
            tree: str = None,  # MDSplus tree or None for PTDATA
            max_sample_rate: int = None,  # Hz
    ) -> dict:
        try:
            if tree:
                self.mdsplus_connection.openTree(tree, self.shot)
                data = np.array(self.mdsplus_connection.get(f'\\{node_name}'), dtype=np.float32)
                time = np.array(self.mdsplus_connection.get(f'dim_of(\\{node_name})'), dtype=np.float32)
                self.mdsplus_connection.closeTree(tree, self.shot)
            else:
                ptdata = f'_n = ptdata("{node_name}", {self.shot})'
                data = np.array(self.mdsplus_connection.get(ptdata), dtype=np.float32)
                time = np.array(self.mdsplus_connection.get('dim_of(_n)'), dtype=np.float32)
        except Exception as e:
            assert False, f'Shot {self.shot} bad data for node {node_name} ({e.__class__.__name__})'
        if data.size < time.size:
            time = time[:data.size]
        if max_sample_rate:
            rate = time.size / (time[-1]-time[0]) * 1e3  # sample rate in Hz
            downsample_factor = int(rate / max_sample_rate)
            if downsample_factor >= 2:
                data = data[::downsample_factor]
                time = time[::downsample_factor]
        assert isinstance(self.bes_time, np.ndarray) and self.bes_time.size > 0
        time_mask = np.logical_and(
            time >= self.bes_time[0],
            time <= self.bes_time[-1],
        )
        data = data[time_mask]
        time = time[time_mask]
        return {
            'point_name': node_name,
            'data': data,
            'time': time,
        }

    def print_contents(self):
        for attr_name in dir(self):
            if attr_name.startswith('_'): continue
            attr = getattr(self, attr_name)
            if attr.__class__.__name__ == 'method': continue
            line = f'  {attr_name}: class {attr.__class__.__name__}'
            if isinstance(attr, Iterable):
                line += f'  size {len(attr)}'
            if isinstance(attr, np.ndarray) and attr.size > 1:
                line += f'  shape {attr.shape}  dtype {attr.dtype}  memsize {sys.getsizeof(attr)/1024/1024:.1f} MB'
            if attr.__class__.__name__.startswith(('int','float')):
                line += f'  value {attr:.3f}'
            elif attr.__class__.__name__.startswith('bool'):
                line += f'  value {attr}'
            print(line)


@dataclasses.dataclass
class HDF5_Data:
    hdf5_file: str|Path = './metadata_default.hdf5'
    truncate_hdf5: bool = False
    really_truncate_hdf5: bool = False

    def __post_init__(self) -> None:
        self.hdf5_file = Path(self.hdf5_file).absolute()
        self.truncate_hdf5 = self.truncate_hdf5 and self.really_truncate_hdf5
        print(f'HDF5 data file: {self.hdf5_file}')
        print('Truncating data file' if self.truncate_hdf5 else 'Appending data file')
        assert self.hdf5_file.parent.exists()
        with h5py.File(self.hdf5_file, 'w' if self.truncate_hdf5 else 'a') as root:
            root.require_group('shots')
            root.require_group('elms')
            root.require_group('configurations')
            root['configurations'].require_group('8x8_configurations')
            root['configurations'].require_group('non_8x8_configurations')
        self.shotlist = []

    def create_metadata_file(
        self,
        csv_file: str|Path = '',
        shotlist: Iterable[int]|np.ndarray = (162303,183781,193757,196560),
        max_shots: int = None,
        bes_channels: Iterable[int] = None,
        max_bes_sample_rate: float = None,  # freq in Hz
        with_limited_signals: bool = False,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        only_pos_ip: bool = False,
        only_neg_bt: bool = False,
        min_pinj_15l: float = None,  # power in W
        min_sustained_15l: float = None,  # time in ms
        use_concurrent: bool = False,
        max_workers: int = None,
    ) -> None:
        if csv_file:
            shotlist = self._read_shotlist(csv_file=csv_file)
        shotlist = np.array(shotlist, dtype=int)
        if max_shots and shotlist.size > max_shots:
            shotlist = shotlist[:max_shots]
        print(f"Shots for metadata file: {shotlist.size}")
        self._loop_over_shotlist(
            shotlist=shotlist,
            bes_channels=bes_channels,
            max_bes_sample_rate=max_bes_sample_rate,
            with_limited_signals=with_limited_signals,
            skip_metadata=False,
            only_8x8=only_8x8,
            only_standard_8x8=only_standard_8x8,
            only_neg_bt=only_neg_bt,
            only_pos_ip=only_pos_ip,
            min_pinj_15l=min_pinj_15l,
            min_sustained_15l=min_sustained_15l,
            use_concurrent=use_concurrent,
            max_workers=max_workers,
        )
        with h5py.File(self.hdf5_file, 'a') as h5root:
            n_shots = n_8x8_shots = n_standard_8x8_shots = n_non_8x8_shots = 0
            n_pos_ip_pos_bt = n_pos_ip_neg_bt = n_neg_ip_pos_bt = n_neg_ip_neg_bt = 0
            for shot_str in h5root['shots']:
                shot_group = h5root['shots'][shot_str]
                n_shots += 1
                if shot_group.attrs['is_8x8']:
                    n_8x8_shots += 1
                else:
                    n_non_8x8_shots += 1
                if shot_group.attrs['is_standard_8x8']:
                    n_standard_8x8_shots += 1
                if shot_group.attrs['ip_pos_phi']:
                    if shot_group.attrs['bt_pos_phi']:
                        n_pos_ip_pos_bt += 1
                    else:
                        n_pos_ip_neg_bt += 1
                else:
                    if shot_group.attrs['bt_pos_phi']:
                        n_neg_ip_pos_bt += 1
                    else:
                        n_neg_ip_neg_bt += 1
            assert n_shots == n_8x8_shots + n_non_8x8_shots
            assert n_shots == n_pos_ip_pos_bt + n_pos_ip_neg_bt + n_neg_ip_pos_bt + n_neg_ip_neg_bt
            h5root.attrs['n_shots'] = n_shots
            h5root.attrs['n_8x8_shots'] = n_8x8_shots
            h5root.attrs['n_non_8x8_shots'] = n_non_8x8_shots
            h5root.attrs['n_standard_8x8_shots'] = n_standard_8x8_shots
            h5root.attrs['n_pos_ip_pos_bt'] = n_pos_ip_pos_bt
            h5root.attrs['n_pos_ip_neg_bt'] = n_pos_ip_neg_bt
            h5root.attrs['n_neg_ip_pos_bt'] = n_neg_ip_pos_bt
            h5root.attrs['n_neg_ip_neg_bt'] = n_neg_ip_neg_bt
            group_8x8 = h5root['configurations']['8x8_configurations']
            group_non_8x8 = h5root['configurations']['non_8x8_configurations']
            for config_group in [group_8x8, group_non_8x8]:
                n_shots = 0
                for config in config_group.values():
                    n_shots += config.attrs['n_shots']
                if 'non' in config_group.name:
                    assert n_shots == h5root.attrs['n_non_8x8_shots']
                else:
                    assert n_shots == h5root.attrs['n_8x8_shots']
                config_group.attrs['n_shots'] = n_shots

    def _read_shotlist(self, csv_file: str|Path) -> np.ndarray:
        csv_file = Path(csv_file).absolute()
        print(f'Reading shotlist from {csv_file}')
        assert csv_file.exists()
        with csv_file.open() as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            assert 'shot' in reader.fieldnames
            shotlist = [int(row['shot']) for row in reader]
        shotlist = np.array(shotlist, dtype=int)
        print(f"Shots in shotlist: {shotlist.size}")
        assert shotlist.size > 0
        return shotlist

    def _loop_over_shotlist(
        self,
        shotlist: Iterable[int],
        bes_channels: Iterable[int] = None,
        max_bes_sample_rate: float = None,  # freq in Hz
        with_limited_signals: bool = False,
        skip_metadata: bool = False,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        only_pos_ip: bool = False,
        only_neg_bt: bool = False,
        min_pinj_15l: float = None,  # power in W
        min_sustained_15l: float = None,  # time in ms
        use_concurrent: bool = False,
        max_workers: int = None,
    ) -> None:
        with h5py.File(self.hdf5_file, 'a') as h5root:
            only_8x8 = only_8x8 or only_standard_8x8
            mdsplus_connection = make_mdsplus_connection()
            if use_concurrent:
                if not max_workers:
                    max_workers = len(os.sched_getaffinity(0))
                lock = threading.Lock()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for shot in shotlist:
                        futures.append(
                            executor.submit(
                                self._load_shot_data,
                                shot=shot,
                                h5root=h5root,
                                bes_channels=bes_channels,
                                max_bes_sample_rate=max_bes_sample_rate,
                                with_limited_signals=with_limited_signals,
                                skip_metadata=skip_metadata,
                                only_8x8=only_8x8,
                                only_standard_8x8=only_standard_8x8,
                                only_neg_bt=only_neg_bt,
                                only_pos_ip=only_pos_ip,
                                min_pinj_15l=min_pinj_15l,
                                min_sustained_15l=min_sustained_15l,
                                mdsplus_connection=mdsplus_connection,
                                lock=lock,
                            )
                        )
                    while True:
                        running = 0
                        done = 0
                        valid_shot_count = 0
                        for future in futures:
                            running += int(future.running())
                            done += int(future.done())
                            if future.done():
                                if (future.result() is not None) and (future.exception() is None):
                                    valid_shot_count += 1
                        with open('futures_status.txt', 'w') as txt_file:
                            txt_file.write(f"{datetime.now()}\n")
                            txt_file.write(f"Total futures: {len(futures)}\n")
                            txt_file.write(f"Running: {running}\n")
                            txt_file.write(f"Done: {done}\n")
                            txt_file.write(f"Valid shots: {valid_shot_count}\n")
                        if running == 0:
                            break
                        time.sleep(60)
            else:
                for i, shot in enumerate(shotlist):
                    print(f'Shot {shot} ({i + 1} of {shotlist.size})')
                    self._load_shot_data(
                        shot=shot,
                        h5root=h5root,
                        bes_channels=bes_channels,
                        max_bes_sample_rate=max_bes_sample_rate,
                        with_limited_signals=with_limited_signals,
                        skip_metadata=skip_metadata,
                        only_8x8=only_8x8,
                        only_standard_8x8=only_standard_8x8,
                        only_neg_bt=only_neg_bt,
                        only_pos_ip=only_pos_ip,
                        min_pinj_15l=min_pinj_15l,
                        min_sustained_15l=min_sustained_15l,
                        mdsplus_connection=mdsplus_connection,
                    )

    def _load_shot_data(
        self,
        shot: int,
        h5root: h5py.File,
        bes_channels: Iterable = None,
        max_bes_sample_rate: float = None,  # freq in Hz
        with_limited_signals: bool = False,
        skip_metadata: bool = False,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        only_pos_ip: bool = False,
        only_neg_bt: bool = False,
        min_pinj_15l: float = None,  # power in W
        min_sustained_15l: float = None,  # time in ms
        mdsplus_connection = None,
        lock: threading.Lock = None,
    ) -> int:
        try:
            bes_data = Shot_Data(
                shot=shot,
                bes_channels=bes_channels,
                max_bes_sample_rate=max_bes_sample_rate,
                with_limited_signals=with_limited_signals,
                skip_metadata=skip_metadata,
                only_8x8=only_8x8,
                only_standard_8x8=only_standard_8x8,
                only_pos_ip=only_pos_ip,
                only_neg_bt=only_neg_bt,
                min_pinj_15l=min_pinj_15l,
                min_sustained_15l=min_sustained_15l,
                mdsplus_connection=mdsplus_connection,
                # quiet=True,
            )
        except Exception as e:
            print(f"Shot {shot} failed: {e}")
            return
        assert 'configurations' in h5root
        assert '8x8_configurations' in h5root['configurations']
        assert 'non_8x8_configurations' in h5root['configurations']
        group_8x8 = h5root['configurations']['8x8_configurations']
        group_non_8x8 = h5root['configurations']['non_8x8_configurations']
        if lock is None:
            lock = contextlib.nullcontext()
        with lock:
            config_index = None
            config_index_count = [0, 1000]
            # check for match to existing configuration
            for igroup, config_group in enumerate([group_8x8, group_non_8x8]):
                for config_index_str, config in config_group.items():
                    config: h5py.Group
                    is_same = False
                    config_index_count[igroup] = np.max(
                        [config_index_count[igroup], int(config_index_str)]
                    )
                    for position in ['r_position','z_position']:
                        is_same = np.allclose(
                            getattr(bes_data, position),
                            config.attrs[position],
                            atol=0.2,
                        )
                        if not is_same:
                            break
                    if is_same:
                        if shot not in config.attrs['shots']:
                            config.attrs['shots'] = np.append(config.attrs['shots'], shot)
                            config.attrs['n_shots'] = config.attrs['shots'].size
                        config_index = int(config_index_str)
                        break
                if config_index:
                    break
            # create new configuration group
            if config_index is None:
                config_index = config_index_count[0 if bes_data.is_8x8 else 1] + 1
                config_group = group_8x8 if bes_data.is_8x8 else group_non_8x8
                new_config = config_group.create_group(f'{config_index:04d}')
                new_config.attrs['shots'] = np.array([bes_data.shot], dtype=int)
                new_config.attrs['n_shots'] = new_config.attrs['shots'].size
                for key in ['r_position','z_position','is_8x8','is_standard_8x8']:
                    new_config.attrs[key] = getattr(bes_data, key)
                if bes_data.is_8x8:
                    for key in ['r_avg','z_avg','inboard_column_channel_order','delz_avg']:
                        new_config.attrs[key] = getattr(bes_data, key)
            assert config_index
            # save shot metadata
            shot_group = h5root['shots'].require_group(str(shot))
            shot_group.attrs['configuration_index'] = config_index
            for attr_name in dir(bes_data):
                if attr_name.startswith('_'): continue
                attr = getattr(bes_data, attr_name)
                if attr.__class__.__name__ == 'method': continue
                if isinstance(attr, MDSplus.Connection): continue
                if attr is None: attr = False
                if isinstance(attr, np.ndarray) and attr.size > 100:
                    shot_group.require_dataset(
                        attr_name,
                        data=attr,
                        shape=attr.shape,
                        dtype=attr.dtype,
                    )
                else:
                    shot_group.attrs[attr_name] = attr
        return shot

    def append_elm_event_data(self): pass

    def filter_shots(
            self,
            r_avg=None,
            z_avg=None,
            delz_avg=None,
            ip_extremum=None,
            bt_extremum=None,
            pinj_15l_max=None,
            pinj_15r_max=None,
            only_standard_8x8=False,
            export_csv=True,
            filename_prefix='filtered',
    ) -> list:
        shotlist = []
        assert self.hdf5_file.exists()
        inputs = {
            'r_avg': r_avg,
            'z_avg': z_avg,
            'delz_avg': delz_avg,
            'ip_extremum': ip_extremum,
            'bt_extremum': bt_extremum,
            'pinj_15l_max': pinj_15l_max,
            'pinj_15r_max': pinj_15r_max,
        }
        candidate_shots = 0
        with h5py.File(self.hdf5_file, 'r') as root:
            input_violation_counts = {key: 0 for key in inputs}
            for shot_key in root['shots']:
                if only_standard_8x8 and not shot_group.attrs['is_standard_8x8']:
                    continue
                shot_group = root['shots'][shot_key]
                candidate_shots += 1
                in_range = True
                for input_key, input_value in inputs.items():
                    if input_value is not None and len(input_value)==2:
                        if (shot_group.attrs[input_key] < input_value[0] or
                            shot_group.attrs[input_key] > input_value[1]):
                            in_range = False
                            input_violation_counts[input_key] += 1
                if in_range:
                    shotlist.append(shot_group.attrs['shot'])
        print(f'Valid shots with filter: {len(shotlist)} (total candidate shots {candidate_shots})')
        for input_key, count in input_violation_counts.items():
            print(f"  {input_key} violations: {count}  (range {inputs[input_key]})")
        if export_csv:
            filename = f"{filename_prefix}_shotlist.csv"
            print(f"Creating CSV shotlist: {filename}")
            with open(filename, 'w') as csvfile:
                fields = ['shot', 'start_time', 'stop_time']
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                with h5py.File(self.hdf5_file, 'r') as root:
                    shotlist_data = [
                        {key: root['shots'][str(shot)].attrs[key] for key in fields}
                        for shot in shotlist
                    ]
                writer.writerows(shotlist_data)
        return shotlist

    def plot_ip_bt_histograms(self, filename='', shotlist=tuple()):
        with h5py.File(self.hdf5_file, 'r') as root:
            ip = np.zeros(len(root['shots'])) * np.NAN
            bt = np.zeros(len(root['shots'])) * np.NAN
            for i_key, shot_key in enumerate(root['shots']):
                if shot_key.startswith('config'): continue
                if shotlist and int(shot_key) not in shotlist: continue
                shot_group = root['shots'][shot_key]
                ip[i_key] = shot_group.attrs['ip_extremum']
                bt[i_key] = shot_group.attrs['bt_extremum']
        ip = ip[np.isfinite(ip)]
        bt = bt[np.isfinite(bt)]
        plt.figure(figsize=(6,3))
        plt.subplot(121)
        plt.hist(ip/1e3, bins=21)
        plt.xlabel('Ip (kA)')
        plt.subplot(122)
        plt.hist(bt, bins=21)
        plt.xlabel('Bt (T)')
        for axes in plt.gcf().axes:
            axes.set_ylabel('Shot count')
            axes.set_title(f'Shots: {ip.size}')
        plt.tight_layout()
        plt.savefig(filename if filename else 'ip_bt_hist.pdf', format='pdf')

    def plot_8x8_rz_avg(self, filename='', shotlist=tuple()):
        with h5py.File(self.hdf5_file, 'r') as root:
            r_avg = np.zeros(len(root['shots'])) * np.NAN
            z_avg = np.zeros(len(root['shots'])) * np.NAN
            for i_key, shot_key in enumerate(root['shots']):
                if shot_key.startswith('config'): continue
                if shotlist and int(shot_key) not in shotlist: continue
                shot_group = root['shots'][shot_key]
                r_avg[i_key] = shot_group.attrs['r_avg']
                z_avg[i_key] = shot_group.attrs['z_avg']
        r_avg = r_avg[np.isfinite(r_avg)]
        z_avg = z_avg[np.isfinite(z_avg)]
        plt.figure(figsize=(4,4))
        plt.subplot(111)
        plt.plot(r_avg, z_avg, 'xb', alpha=0.5)
        plt.xlim(218, 232)
        plt.xlabel('R (cm)')
        plt.ylim(-8, 8)
        plt.ylabel('Z (cm)')
        plt.title(f'Shots: {r_avg.size}')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(filename if filename else '8x8_configurations.pdf', format='pdf')

    def plot_configurations(self) -> None:
        with h5py.File(self.hdf5_file, 'r') as root:
            config_group = root['configurations']['8x8_configurations']
            assert len(config_group) > 0
            _, axes = plt.subplots(nrows=4, ncols=3, figsize=(8.5,11))
            i_page = 0
            for i_config, (key, group) in enumerate(config_group.items()):
                plot_axes = i_config%axes.size
                if plot_axes == 0:
                    for ax in axes.flat:
                        ax.clear()
                plt.sca(axes.flat[plot_axes])
                plt.plot(
                    group.attrs['r_position'], 
                    group.attrs['z_position'],
                    'xb',
                    markersize=3,
                )
                plt.title(
                    f"Config {key} | n_shots {group.attrs['n_shots']}",
                    fontsize='medium',
                )
                plt.xlim(205, 235)
                plt.xlabel('R (cm)')
                plt.ylim(-10, 10)
                plt.ylabel('Z (cm)')
                plt.annotate(f"R_avg={group.attrs['r_avg']:.1f}", 
                             xy=[0.05,0.12], xycoords='axes fraction', fontsize='x-small')
                plt.annotate(f"Z_avg={group.attrs['z_avg']:.1f}",
                             xy=[0.05,0.03], xycoords='axes fraction', fontsize='x-small')
                plt.gca().set_aspect('equal')
                if plot_axes == axes.size-1 or i_config == len(config_group)-1:
                    i_page += 1
                    plt.tight_layout()
                    plt.savefig(f'configs_{i_page:02d}.pdf', format='pdf')

    def print_hdf5_contents(
            self, 
            print_attributes: bool = True,
            print_datasets: bool = True,
    ):
        print_hdf5_contents(
            self.hdf5_file, 
            print_attributes=print_attributes,
            print_datasets=print_datasets,
        )
    

if __name__=='__main__':
    # bes_data = Shot_Data(
    #     bes_channels=range(1,3),
    #     with_limited_signals=True,
    # )
    # bes_data.print_contents()

    dataset = HDF5_Data(truncate_hdf5=True, really_truncate_hdf5=True)
    dataset.create_metadata_file(bes_channels=[23], with_limited_signals=True, max_bes_sample_rate=200e3)
    dataset.print_hdf5_contents()
