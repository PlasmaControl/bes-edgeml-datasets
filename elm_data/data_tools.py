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
class Shot:
    shot: int = 196560
    channels: Iterable[int] = None
    with_other_signals: bool = False
    only_8x8: bool = False
    only_standard_8x8: bool = False
    only_pos_ip: bool = False
    only_neg_bt: bool = False
    min_pinj_15l: float = None
    min_sustained_15l: float = None
    connection: MDSplus.Connection = None
    quiet: bool = False

    def __post_init__(self):
        t1 = time.time()
        if self.connection is None:
            self.connection = make_mdsplus_connection()
        if self.channels is None:
            self.channels = []
        self.channels = np.array(self.channels, dtype=int)
        self.bes_signals = None
        # self.metadata = None
        self.is_8x8 = self.is_standard_8x8 = False
        self.only_8x8 = self.only_8x8 or self.only_standard_8x8
        # get BES time array
        self.bes_time = np.array(
            self.connection.get(f'dim_of(ptdata("besfu01", {self.shot}))'),
            dtype=np.float32,
        )
        assert self.bes_time.size > 0, f'Shot {self.shot} has bad BES time data'
        self.start_time = self.bes_time[0]
        self.stop_time = self.bes_time[-1]
        # get BES configuration metadata
        self.connection.openTree('bes', self.shot)
        self.r_position = np.array(self.connection.get(r'\bes_r'), dtype=np.float32)
        self.z_position = -np.array(self.connection.get(r'\bes_z'), dtype=np.float32)
        assert self.r_position.size == 64 and self.z_position.size == 64, f'Shot {self.shot} has bad BES position data'
        self.connection.closeTree('bes', self.shot)
        for i in np.arange(8):
            # del-r of column i
            rdiff = np.diff(self.r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.12)
            # del-z of row i
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
            result = self.get_signal(node_name, max_sample_rate=5e3)
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
        # get BES signals
        if self.channels.size > 0:
            self.get_bes_signals()
        # mask for early Ip termination
        ip_mask = np.flatnonzero(self.ip >= 400e3)
        ip_stop_time = self.ip_time[ip_mask[-1]]
        if ip_stop_time - 100 < self.stop_time:
            bes_mask = self.bes_time <= ip_stop_time-100
            self.bes_time = self.bes_time[bes_mask]
            if self.bes_signals is not None:
                # new_signals = self.bes_signals[:, bes_mask]
                self.bes_signals = self.bes_signals[:, bes_mask]
            self.start_time = self.bes_time[0]
            self.stop_time = self.bes_time[-1]
            for node_name in ['ip', 'bt']:
                signal = getattr(self, node_name)
                signal_time = getattr(self, f'{node_name}_time')
                mask = signal_time <= self.stop_time
                setattr(self, node_name, signal[mask])
                setattr(self, f'{node_name}_time', signal_time[mask])
        # get NB
        for node_name in ['pinj', 'pinj_15l', 'pinj_15r']:
            result = self.get_signal(node_name, tree='nb', max_sample_rate=5e3)
            setattr(self, node_name, result['data'])
            setattr(self, f'{node_name}_max', result['data'].max())
            if node_name == 'pinj':
                setattr(self, f'{node_name}_time', result['time'])
                self.connection.openTree('nb', self.shot)
                date = self.connection.get(f'getnci(\\pinj, "time_inserted")')
                self.date = str(date.date)
                self.connection.closeTree('nb', self.shot)
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
        # get other signals
        if self.with_other_signals:
            self.get_other_siganls()
        if self.bes_signals is None:
            self.bes_time = None

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

    def get_other_siganls(self):
        t1 = time.time()
        node_names = ['FS03', 'FS04', 'FS05']
        for i_node, node_name in enumerate(node_names):
            signal_dict = self.get_signal(
                node_name=node_name,
                tree='spectroscopy',
                max_sample_rate=50e3,
            )
            setattr(self, node_name, signal_dict['data'])
            if i_node==0:
                setattr(self, 'FS_time', signal_dict['time'])
        node_names = ['denv3f']
        for i_node, node_name in enumerate(node_names):
            signal_dict = self.get_signal(
                node_name=node_name,
                tree='electrons',
                max_sample_rate=200e3,
            )
            setattr(self, node_name, signal_dict['data'])
            if i_node==0:
                setattr(self, f'{node_name}_time', signal_dict['time'])
        if not self.quiet: print(f'{self.shot}: Non-BES signal time = {time.time() - t1:.2f} s')

    def get_bes_signals(
            self, 
            channels: Iterable = None, 
            max_sample_rate: float = 200,
    ):
        self.channels = np.array(channels, dtype=int) if channels else self.channels
        assert np.all(self.channels > 0), f'Shot {self.shot}: BES does not have channel `0`'
        self.bes_signals = np.empty([self.channels.size, self.bes_time.size], dtype=np.float32)
        t1 = time.time()
        tdi_vars = []
        tdi_assignments = []
        for channel in self.channels:
            var = f'_n{channel:02d}_{self.shot}'
            tdi_vars.append(var)
            tmp = f'{var} = ptdata("besfu{channel:02d}", {self.shot})'
            tdi_assignments.append(tmp)
        self.connection.get(', '.join(tdi_assignments))
        for i, tdi_var in enumerate(tdi_vars):
            self.bes_signals[i, :] = np.array(self.connection.get(tdi_var), dtype=np.float32)
        if max_sample_rate:
            sample_rate = 1 / np.mean(np.diff(self.bes_time[:1000]))  # kHz
            downsample_factor = int(np.rint(sample_rate / max_sample_rate))
            if downsample_factor >= 2:
                self.bes_signals = self.bes_signals[:, ::downsample_factor]
                self.bes_time = self.bes_time[::downsample_factor]
        if not self.quiet: print(f'{self.shot}: BES signal time = {time.time() - t1:.2f} s')

    def get_signal(
            self,
            node_name: str,
            tree: str = None,  # MDSplus tree or None for PTDATA
            max_sample_rate: int = None,  # max sample rate in Hz (approx)
    ) -> dict:
        try:
            if tree:
                self.connection.openTree(tree, self.shot)
                data = np.array(self.connection.get(f'\\{node_name}'), dtype=np.float32)
                time = np.array(self.connection.get(f'dim_of(\\{node_name})'), dtype=np.float32)
                self.connection.closeTree(tree, self.shot)
            else:
                ptdata = f'_n = ptdata("{node_name}", {self.shot})'
                data = np.array(self.connection.get(ptdata), dtype=np.float32)
                time = np.array(self.connection.get('dim_of(_n)'), dtype=np.float32)
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
        if self.bes_time is not None:
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


@dataclasses.dataclass
class HDF5_Data:
    hdf5_file: str|Path = 'test_metadata.hdf5'

    def __post_init__(self) -> None:
        self.hdf5_file = Path(self.hdf5_file)

    def load_shotlist(
        self,
        csv_file: str|Path = '',
        shotlist: Iterable[int] = (162303,183781,193757,196560),
        truncate_hdf5: bool = False,
        max_shots: int = None,
        use_concurrent: bool = False,
        max_workers: int = None,
        channels: Iterable[int] = None,
        with_other_signals: bool = False,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        only_pos_ip: bool = False,
        only_neg_bt: bool = False,
        min_pinj_15l: float = None,  # power in W
        min_sustained_15l: float = None,  # time in ms
    ) -> None:
        if csv_file:
            # read shotlist from CSV file; column `shot` must exist
            csv_file = Path(csv_file)
            assert csv_file.exists()
            print(f'Using shotlist {csv_file.as_posix()}')
            shotlist = []
            with csv_file.open() as csvfile:
                reader = csv.DictReader(csvfile, skipinitialspace=True)
                assert 'shot' in reader.fieldnames
                for irow, row in enumerate(reader):
                    if max_shots and irow+1 > max_shots: break
                    shotlist.append(int(row['shot']))
        shotlist = np.array(shotlist, dtype=int)
        print(f"Shots in shotlist: {shotlist.size}")
        assert shotlist.size > 0
        only_8x8 = only_8x8 or only_standard_8x8
        h5_mode = 'w' if truncate_hdf5 else 'a'
        with h5py.File(self.hdf5_file, h5_mode) as h5root:
            valid_shot_count = 0
            configuration_group = h5root.require_group('configurations')
            group_8x8 = configuration_group.require_group('8x8_configurations')
            group_non_8x8 = configuration_group.require_group('non_8x8_configurations')
            connection = make_mdsplus_connection()
            if use_concurrent:
                if not max_workers:
                    max_workers = len(os.sched_getaffinity(0))
                lock = threading.Lock()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for shot in shotlist:
                        futures.append(
                            executor.submit(
                                self._check_channel_configuration,
                                shot=shot,
                                h5root=h5root,
                                group_8x8=group_8x8,
                                group_non_8x8=group_non_8x8,
                                channels=channels,
                                with_other_signals=with_other_signals,
                                only_8x8=only_8x8,
                                only_standard_8x8=only_standard_8x8,
                                lock=lock,
                                connection=connection,
                                only_neg_bt=only_neg_bt,
                                only_pos_ip=only_pos_ip,
                                min_pinj_15l=min_pinj_15l,
                                min_sustained_15l=min_sustained_15l,
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
                    result = self._check_channel_configuration(
                        shot=shot,
                        h5root=h5root,
                        group_8x8=group_8x8,
                        group_non_8x8=group_non_8x8,
                        channels=channels,
                        with_other_signals=with_other_signals,
                        only_8x8=only_8x8,
                        only_standard_8x8=only_standard_8x8,
                        connection=connection,
                        only_neg_bt=only_neg_bt,
                        only_pos_ip=only_pos_ip,
                        min_pinj_15l=min_pinj_15l,
                        min_sustained_15l=min_sustained_15l,
                    )
                    if result:
                        valid_shot_count += 1
            h5root.flush()
            # count shots
            n_shots = n_8x8_shots = n_standard_8x8_shots = n_non_8x8_shots = 0
            n_pos_ip_pos_bt = n_pos_ip_neg_bt = n_neg_ip_pos_bt = n_neg_ip_neg_bt = 0
            for group_name, group in h5root.items():
                if group_name.startswith('config'):
                    continue
                n_shots += 1
                if group.attrs['is_8x8']:
                    n_8x8_shots += 1
                else:
                    n_non_8x8_shots += 1
                if group.attrs['is_standard_8x8']:
                    n_standard_8x8_shots += 1
                if group.attrs['ip_pos_phi'] is True:
                    if group.attrs['bt_pos_phi'] is True:
                        n_pos_ip_pos_bt += 1
                    else:
                        n_pos_ip_neg_bt += 1
                else:
                    if group.attrs['bt_pos_phi'] is True:
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
            h5root.flush()
            for config_group in [group_8x8, group_non_8x8]:
                n_shots = 0
                for config in config_group.values():
                    n_shots += config.attrs['n_shots']
                if 'non' in config_group.name:
                    assert n_shots == h5root.attrs['n_non_8x8_shots']
                else:
                    assert n_shots == h5root.attrs['n_8x8_shots']
                config_group.attrs['n_shots'] = n_shots
    
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
        with h5py.File(self.hdf5_file, 'r') as hfile:
            input_violation_counts = {key: 0 for key in inputs}
            for shot_key, shot_group in hfile.items():
                if shot_key.startswith('config'):
                    continue
                if only_standard_8x8 and not shot_group.attrs['is_standard_8x8']:
                    continue
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
                with h5py.File(self.hdf5_file, 'r') as hfile:
                    shotlist_data = [
                        {key: hfile[f'{shot}'].attrs[key] for key in fields}
                        for shot in shotlist
                    ]
                writer.writerows(shotlist_data)
        return shotlist

    def print_hdf5_contents(self):
        print(f'Contents of {self.hdf5_file}')
        with h5py.File(self.hdf5_file, 'r') as h5file:
            self._recursively_print_content(h5file)

    def print_hdf5_summary(self):
        print(f'Summary of {self.hdf5_file}')
        with h5py.File(self.hdf5_file, 'r') as h5file:
            print(f"Group {h5file.name}")
            self._print_attributes(h5file)
            config_group = h5file['configurations']
            for group in config_group.values():
                print(f"Group {group.name}")
                self._print_attributes(group)

    def plot_ip_bt_histograms(self, filename='', shotlist=tuple()):
        with h5py.File(self.hdf5_file, 'r') as hfile:
            ip = np.zeros(len(hfile)) * np.NAN
            bt = np.zeros(len(hfile)) * np.NAN
            for i_key, (shot_key, shot_group) in enumerate(hfile.items()):
                if shot_key.startswith('config'): continue
                if shotlist and int(shot_key) not in shotlist: continue
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
        with h5py.File(self.hdf5_file, 'r') as hfile:
            r_avg = np.zeros(len(hfile)) * np.NAN
            z_avg = np.zeros(len(hfile)) * np.NAN
            for i_key, (shot_key, shot_group) in enumerate(hfile.items()):
                if shot_key.startswith('config'): continue
                if shotlist and int(shot_key) not in shotlist: continue
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
        with h5py.File(self.hdf5_file, 'r') as hfile:
            config_group = hfile['configurations']['8x8_configurations']
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

    def _check_channel_configuration(
        self,
        shot: int,
        h5root: h5py.File,
        group_8x8: h5py.Group,
        group_non_8x8: h5py.Group,
        channels: Iterable = None,
        with_other_signals: bool = False,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        only_pos_ip=True,
        only_neg_bt=True,
        min_pinj_15l=700e3,
        min_sustained_15l=300,
        lock: threading.Lock = None,
        connection = None,
    ) -> int:
        try:
            bes_data = Shot(
                channels=channels,
                with_other_signals=with_other_signals,
                connection=connection,
                shot=shot,
                only_8x8=only_8x8,
                only_standard_8x8=only_standard_8x8,
                only_pos_ip=only_pos_ip,
                only_neg_bt=only_neg_bt,
                min_pinj_15l=min_pinj_15l,
                min_sustained_15l=min_sustained_15l,
                quiet=True,
            )
        except Exception as e:
            print(f"Shot {shot} failed: {e}")
            return None
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
                if bes_data.is_8x8:
                    config_index = config_index_count[0] + 1
                    new_config = group_8x8.create_group(f'{config_index:04d}')
                else:
                    config_index = config_index_count[1] + 1
                    new_config = group_non_8x8.create_group(f'{config_index:04d}')
                new_config.attrs['shots'] = np.array([bes_data.shot], dtype=int)
                new_config.attrs['n_shots'] = new_config.attrs['shots'].size
                for key in ['r_position','z_position','is_8x8','is_standard_8x8']:
                    new_config.attrs[key] = getattr(bes_data, key)
                if bes_data.is_8x8:
                    for key in ['r_avg','z_avg','inboard_column_channel_order','delz_avg']:
                        new_config.attrs[key] = getattr(bes_data, key)
            assert config_index
            # save shot metadata
            shot_group = h5root.require_group(str(shot))
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
            h5root.flush()
        return shot
    
    @staticmethod
    def _print_attributes(obj: h5py.Group|h5py.Dataset):
        for key, value in obj.attrs.items():
            if isinstance(value, np.ndarray):
                print(f'  Attribute {key}:', value.shape, value.dtype)
            else:
                print(f'  Attribute {key}:', value)

    def _recursively_print_content(self, group: h5py.Group):
        print(f'Group {group.name}')
        self._print_attributes(group)
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                self._recursively_print_content(value)
            if isinstance(value, h5py.Dataset):
                print(f'  Dataset {key}:', value.shape, value.dtype)
                self._print_attributes(value)


if __name__=='__main__':
    # bes_data = Shot(
    #     channels=np.arange(2)+1,
    #     with_other_signals=True,
    #     min_sustained_15l=300,
    # )
    # bes_data.print_contents()

    dataset = HDF5_Data()
    dataset.load_shotlist(truncate_hdf5=True, channels=[23], with_other_signals=True)
    dataset.print_hdf5_contents()

    # dataset = HDF5_Data(
    #     # hdf5_file='/home/smithdr/ml/elm_data/step_4_shot_partial_data/data_v1.hdf5',
    #     hdf5_file='/home/smithdr/ml/elm_data/step_4_shot_partial_data/data_v2.hdf5',
    # )
    # dataset.print_hdf5_contents()
