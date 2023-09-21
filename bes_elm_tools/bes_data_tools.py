import time
from datetime import datetime
from typing import Iterable
import os
import csv
import threading
import concurrent.futures
import contextlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import MDSplus
import h5py


class Shot:

    def __init__(
        self,
        shot: int = 196560,
        channels: Iterable = np.arange(1, 65),
        only_8x8: bool = None,
        only_standard_8x8: bool = None,
        max_delz: float = None,
        only_pos_ip: bool = None,
        only_neg_bt: bool = None,
        min_pinj_15l: float = None,
        connection: MDSplus.Connection = None,
        skip_metadata: bool = False,
    ):
        t1 = time.time()
        self.connection = connection
        if self.connection is None:
            tries = 0
            while True:
                tries += 1
                try:
                    self.connection = MDSplus.Connection('atlas.gat.com')
                    break
                except:
                    time.sleep(5)
                    if tries <= 10:
                        continue
                    raise
        self.shot = shot
        self.channels = np.array(channels, dtype=int)
        self.time = None
        self.signals = None
        self.metadata = None
        self.inboard_column_channel_order = None
        self.is_8x8 = self.is_standard_8x8 = False
        self.delz_avg = None
        self.signal_names = [
            'pinj',
            'pinj_15l',
            'pinj_15r',
        ]
        only_8x8 = only_8x8 or only_standard_8x8
        # get time array
        try:
            sigtime = self.connection.get(f'dim_of(ptdata("besfu01", {self.shot}))')
            self.time = np.array(sigtime)
            # BES configuration metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r'), dtype=np.float32)
            z_position = -np.array(self.connection.get(r'\bes_z'), dtype=np.float32)
            assert r_position.size == 64 and z_position.size == 64
            self.connection.closeTree('bes', self.shot)
        except:
            self.time = None
            print(f'{self.shot}: error with BES data')
            return
        if skip_metadata:
            return
        for i in np.arange(8):
            # del-r of column i
            rdiff = np.diff(r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.12)
            # del-z of row i
            zdiff = np.diff(z_position[i * 8 + np.arange(8)])
            row_test = np.all(np.abs(zdiff) <= 0.12)
            self.is_8x8 = col_test and row_test
            if self.is_8x8 is False:
                break
        if self.is_8x8:
            z_first_column = z_position[np.arange(8) * 8]
            self.inboard_column_channel_order = np.flip(z_first_column.argsort()) * 8
            self.is_standard_8x8 = np.array_equal(
                self.inboard_column_channel_order,
                np.arange(8, dtype=int) * 8,
            )
        if only_8x8 and (self.is_8x8 is False):
            self.time = None
            print(f'{self.shot}: ERROR not 8x8 config')
            return
        if only_standard_8x8 and (self.is_standard_8x8 is False):
            print(f'{self.shot}: ERROR not standard 8x8 config')
            self.time = None
            return
        self.delz_avg = np.abs(np.diff(z_position[np.arange(8)*8]).mean()).round(2)
        if max_delz and self.delz_avg > max_delz:
            self.time = None
            print(f'{self.shot}: exceed max delz')
            return
        self.metadata = {
            'shot': self.shot,
            'start_time': self.time[0],
            'stop_time': self.time[-1],
            'time_units': 'ms',
            'r_position': r_position,
            'z_position': z_position,
            'rz_units': 'cm',
            'is_8x8': self.is_8x8,
            'is_standard_8x8': self.is_standard_8x8,
            'r_avg': np.mean(r_position).round(1) if self.is_8x8 else None,
            'z_avg': np.mean(z_position).round(1) if self.is_8x8 else None,
            'inboard_column_channel_order': self.inboard_column_channel_order if self.is_8x8 else None,
            'delz_avg': self.delz_avg if self.is_8x8 else None,
        }
        # get ip, beams, etc.
        try:
            for node_name in ['ip', 'bt']:
                result = self.get_signal(node_name, max_sample_rate=5e3)
                # if node_name == 'ip':
                #     setattr(self, node_name, result['data'])
                #     setattr(self, f'{node_name}_time', result['time'])
                signal:np.ndarray = result['data']
                self.metadata[node_name] = (
                    signal.max()
                    if np.abs(signal.max()) > np.abs(signal.min())
                    else signal.min()
                )
                if only_pos_ip and node_name=='ip':
                    assert self.metadata[node_name] > 0
                if only_neg_bt and node_name=='bt':
                    assert self.metadata[node_name] < 0
            for node_name in ['pinj', 'pinj_15l', 'pinj_15r']:
                result = self.get_signal(node_name, tree='nb', max_sample_rate=5e3)
                setattr(self, node_name, result['data'])
                self.metadata[node_name] = result['data'].max()
                if min_pinj_15l and node_name=='pinj_15l':
                    assert self.metadata[node_name] > min_pinj_15l
                if node_name == 'pinj':
                    setattr(self, f'{node_name}_time', result['time'])
                    self.connection.openTree('nb', self.shot)
                    date = self.connection.get(f'getnci(\\pinj, "time_inserted")')
                    self.metadata['date'] = str(date.date)
                    self.connection.closeTree('nb', self.shot)
        except:
            self.time = None
            print(f'{self.shot}: ERROR for node {node_name}')
            return
        print(f'{self.shot}: Elapsed time = {time.time() - t1:.2f} s')

    def get_signal(
        self,
        node_name: str,
        tree: str = None,  # MDSplus tree or None for PTDATA
        max_sample_rate: int = None,  # max sample rate in Hz (approx)
    ) -> dict:
        if tree:
            self.connection.openTree(tree, self.shot)
            data = np.array(self.connection.get(f'\\{node_name}'), dtype=np.float32)
            time = np.array(self.connection.get(f'dim_of(\\{node_name})'), dtype=np.float32)
            self.connection.closeTree(tree, self.shot)
        else:
            ptdata = f'_n = ptdata("{node_name}", {self.shot})'
            data = np.array(self.connection.get(ptdata), dtype=np.float32)
            time = np.array(self.connection.get('dim_of(_n)'), dtype=np.float32)
        if data.size < time.size:
            time = time[:data.size]
        if max_sample_rate:
            rate = time.size / (time[-1]-time[0]) * 1e3  # sample rate in Hz
            downsample_factor = int(rate / max_sample_rate)
            if downsample_factor >= 2:
                data = data[::downsample_factor]
                time = time[::downsample_factor]
        if self.time is not None:
            time_mask = np.logical_and(
                time >= self.time[0],
                time <= self.time[-1],
            )
            data = data[time_mask]
            time = time[time_mask]
        return {
            'point_name': node_name,
            'data': data,
            'time': time,
        }

    def get_bes_signals(
        self,
        channels=None,
    ):
        assert self.time is not None
        if channels:
            self.channels = np.array(channels, dtype=int)
        t1 = time.time()
        print(f'{self.shot}: fetching {self.channels.size} BES signals')
        tdi_vars = []
        tdi_assignments = []
        for channel in self.channels:
            var = f'_n{channel:02d}_{self.shot}'
            tdi_vars.append(var)
            tmp = f'{var} = ptdata("besfu{channel:02d}", {self.shot})'
            tdi_assignments.append(tmp)
        self.signals = np.empty([self.channels.size, self.time.size])
        try:
            self.connection.get(', '.join(tdi_assignments))
            for i, tdi_var in enumerate(tdi_vars):
                self.signals[i, :] = self.connection.get(tdi_var)
        except:
            print(f'{self.shot}: ERROR fetching signals')
            self.time = None
            self.signals = None
            return
        print(f'{self.shot}: Signal time = {time.time() - t1:.2f} s')


class HDF5_Data:

    def __init__(
        self,
        hdf5_file: str|Path = 'bes_data.hdf5',
    ) -> None:
        self.hdf5_file = Path(hdf5_file)

    def load_shotlist(
        self,
        csv_file: str|Path = '',
        shotlist: Iterable[int] = (162303,183781,193757,196560),
        truncate_hdf5: bool = False,
        max_shots: int = None,
        use_concurrent: bool = False,
        max_workers: int = None,
        only_8x8: bool = True,
        only_standard_8x8: bool = True,
        max_delz: float = 2.0,
        only_pos_ip=True,
        only_neg_bt=True,
        min_pinj_15l=600e3,
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
                    if max_shots and irow+1 > max_shots:
                        break
                    shotlist.append(int(row['shot']))
        shotlist = np.array(shotlist, dtype=int)
        assert shotlist.size > 0
        only_8x8 = only_8x8 or only_standard_8x8
        h5_mode = 'w' if truncate_hdf5 else 'a'
        with h5py.File(self.hdf5_file, h5_mode) as h5root:
            valid_shot_count = 0
            configuration_group = h5root.require_group('configurations')
            group_8x8 = configuration_group.require_group('8x8_configurations')
            group_non_8x8 = configuration_group.require_group('non_8x8_configurations')
            connection = MDSplus.Connection('atlas.gat.com')
            if use_concurrent:
                if not max_workers:
                    max_workers = len(os.sched_getaffinity(0))
                lock = threading.Lock()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for shot in shotlist:
                        futures.append(
                            executor.submit(
                                self.check_channel_configuration,
                                shot=shot,
                                h5root=h5root,
                                group_8x8=group_8x8,
                                group_non_8x8=group_non_8x8,
                                only_8x8=only_8x8,
                                only_standard_8x8=only_standard_8x8,
                                lock=lock,
                                connection=connection,
                                max_delz=max_delz,
                                only_neg_bt=only_neg_bt,
                                only_pos_ip=only_pos_ip,
                                min_pinj_15l=min_pinj_15l,
                            )
                        )
                    while True:
                        time.sleep(60)
                        running = 0
                        done = 0
                        valid_shot_count = 0
                        for future in futures:
                            running += int(future.running())
                            done += int(future.done())
                            if future.done():
                                if future.result() and future.exception() is None:
                                    valid_shot_count += 1
                        with open('futures_status.txt', 'w') as txt_file:
                            txt_file.write(f"{datetime.now()}\n")
                            txt_file.write(f"Total futures: {len(futures)}\n")
                            txt_file.write(f"Running: {running}\n")
                            txt_file.write(f"Done: {done}\n")
                            txt_file.write(f"Valid shots: {valid_shot_count}\n")
                        if running == 0:
                            break
            else:
                for i, shot in enumerate(shotlist):
                    print(f'Shot {shot} ({i + 1} of {shotlist.size})')
                    result = self.check_channel_configuration(
                        shot=shot,
                        h5root=h5root,
                        group_8x8=group_8x8,
                        group_non_8x8=group_non_8x8,
                        only_8x8=only_8x8,
                        only_standard_8x8=only_standard_8x8,
                        connection=connection,
                        max_delz=max_delz,
                        only_neg_bt=only_neg_bt,
                        only_pos_ip=only_pos_ip,
                        min_pinj_15l=min_pinj_15l,
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
                if group.attrs['ip'] > 0:
                    if group.attrs['bt'] > 0:
                        n_pos_ip_pos_bt += 1
                    else:
                        n_pos_ip_neg_bt += 1
                else:
                    if group.attrs['bt'] > 0:
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
    
    def check_channel_configuration(
        self,
        shot: int,
        h5root: h5py.File,
        group_8x8: h5py.Group,
        group_non_8x8: h5py.Group,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        max_delz: float = None,
        only_pos_ip=True,
        only_neg_bt=True,
        min_pinj_15l=700e3,
        lock: threading.Lock = None,
        connection = None,
    ) -> int:
        bes_data = Shot(
            connection=connection,
            shot=shot,
            only_8x8=only_8x8,
            only_standard_8x8=only_standard_8x8,
            max_delz=max_delz,
            only_pos_ip=only_pos_ip,
            only_neg_bt=only_neg_bt,
            min_pinj_15l=min_pinj_15l,
        )
        if bes_data.time is None:
            del bes_data
            return 0
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
                            bes_data.metadata[position],
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
                if bes_data.metadata['is_8x8']:
                    config_index = config_index_count[0] + 1
                    new_config = group_8x8.create_group(f'{config_index:04d}')
                else:
                    config_index = config_index_count[1] + 1
                    new_config = group_non_8x8.create_group(f'{config_index:04d}')
                new_config.attrs['shots'] = np.array([bes_data.shot], dtype=int)
                new_config.attrs['n_shots'] = new_config.attrs['shots'].size
                for key in ['r_position','z_position','is_8x8','is_standard_8x8']:
                    new_config.attrs[key] = bes_data.metadata[key]
                if bes_data.metadata['is_8x8']:
                    for key in ['r_avg','z_avg','inboard_column_channel_order','delz_avg']:
                        new_config.attrs[key] = bes_data.metadata[key]
            assert config_index
            # save shot metadata
            shot_group = h5root.require_group(str(shot))
            for attr_name, attr_value in bes_data.metadata.items():
                shot_group.attrs[attr_name] = attr_value
                shot_group.attrs['configuration_index'] = config_index
            for point_name in bes_data.signal_names:
                for name in [f'{point_name}', f'{point_name}_time']:
                    data = getattr(bes_data, name, None)
                    if data is None:
                        continue
                    shot_group.require_dataset(
                        name,
                        data=data,
                        shape=data.shape,
                        dtype=data.dtype,
                    )
            h5root.flush()
        return shot
    
    @staticmethod
    def print_attributes(obj) -> None:
        for key, value in obj.attrs.items():
            if isinstance(value, np.ndarray):
                print(f'  Attribute {key}:', value.shape, value.dtype)
            else:
                print(f'  Attribute {key}:', value)

    def recursively_print_content(self, group: h5py.Group) -> None:
        print(f'Group {group.name}')
        self.print_attributes(group)
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                self.recursively_print_content(value)
            if isinstance(value, h5py.Dataset):
                print(f'  Dataset {key}:', value.shape, value.dtype)
                self.print_attributes(value)

    def print_hdf5_contents(self) -> None:
        print(f'Contents of {self.hdf5_file}')
        assert self.hdf5_file.exists()
        with h5py.File(self.hdf5_file, 'r') as h5file:
            self.recursively_print_content(h5file)

    def print_hdf5_summary(self) -> None:
        print(f'Summary of {self.hdf5_file}')
        assert self.hdf5_file.exists()
        with h5py.File(self.hdf5_file, 'r') as h5file:
            print(f"Group {h5file.name}")
            self.print_attributes(h5file)
            config_group = h5file['configurations']
            for group in config_group.values():
                print(f"Group {group.name}")
                self.print_attributes(group)

    def plot_ip_bt_histograms(
            self, 
            filename='ip_bt_hist.pdf',
            **filter_kwargs,
        ) -> None:
        assert self.hdf5_file.exists()
        shotlist = []
        if filter_kwargs:
            shotlist = self.filter(**filter_kwargs)
        with h5py.File(self.hdf5_file, 'r') as hfile:
            ip = np.zeros(len(hfile)) * np.NAN
            bt = np.zeros(len(hfile)) * np.NAN
            for i_key, (shot_key, shot_group) in enumerate(hfile.items()):
                if shot_key.startswith('config'):
                    continue
                if shotlist and int(shot_key) not in shotlist:
                    continue
                ip[i_key] = shot_group.attrs['ip']
                bt[i_key] = shot_group.attrs['bt']
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
        plt.savefig(filename, format='pdf')

    def plot_8x8_rz_avg(
            self, 
            filename='8x8_configurations.pdf',
            **filter_kwargs,
        ) -> None:
        assert self.hdf5_file.exists()
        shotlist = []
        if filter_kwargs:
            shotlist = self.filter(**filter_kwargs)
        with h5py.File(self.hdf5_file, 'r') as hfile:
            r_avg = np.zeros(len(hfile)) * np.NAN
            z_avg = np.zeros(len(hfile)) * np.NAN
            # delz_avg = []
            for i_key, (shot_key, shot_group) in enumerate(hfile.items()):
                if shot_key.startswith('config'):
                    continue
                z_position = shot_group.attrs['z_position']
                # delz = np.abs(np.diff(z_position[np.arange(8)*8]).mean())
                # delz_avg.append(delz)
                if shotlist and int(shot_key) not in shotlist:
                    continue
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
        plt.savefig(filename, format='pdf')

    def plot_configurations(self) -> None:
        assert self.hdf5_file.exists()
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

    def filter(
            self,
            r_avg=None,
            z_avg=None,
            ip=None,
            bt=None,
            pinj_15l=None,
            pinj_15r=None,
            only_standard_8x8=True,
            export_csv=False,
            export_metadata=False,
            export_max_shots=None,
            filename_prefix='filtered',
    ) -> list:
        shotlist = []
        assert self.hdf5_file.exists()
        inputs = {
            'r_avg': r_avg,
            'z_avg': z_avg,
            'ip': ip,
            'bt': bt,
            'pinj_15l': pinj_15l,
            'pinj_15r': pinj_15r,
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
                    if input_value and len(input_value)==2:
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
                if export_max_shots:
                    shotlist_data = shotlist_data[:export_max_shots]
                writer.writerows(shotlist_data)
        if export_metadata:
            pass
        return shotlist

    def save_signals(
        self,
    ) -> None:
        assert self.hdf5_file.exists()
        

if __name__=='__main__':
    # bes_data = BES_Shot_Data()
    # bes_data.get_bes_signals([1,2])

    dataset = HDF5_Data()
    dataset.load_shotlist(truncate_hdf5=True)
    # # dataset.load_shotlist(truncate_hdf5=True, use_concurrent=False, max_workers=2)
    dataset.print_hdf5_contents()
    # dataset.print_hdf5_summary()
    # dataset.plot_ip_bt_histograms()
    # dataset.plot_configurations()
    # dataset.plot_8x8_rz_avg()
    # filter_kwargs = dict(
    #     r_avg = [222.0, 227.5],
    #     z_avg = [-1.5, 1],
    #     ip = [0.6e6, 2e6],
    #     bt = [-3, 0],
    #     pinj_15l = [0.7e6, 2.5e6],
    # )
    # dataset.filter(
    #     filename_prefix='filtered',
    #     export_csv=True,
    #     # export_max_shots=5,
    #     **filter_kwargs,
    # )
    # dataset.plot_8x8_rz_avg(
    #     filename='filtered_config.pdf',
    #     **filter_kwargs,
    # )
