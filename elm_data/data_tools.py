import time
from datetime import datetime
from typing import Iterable
import os
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
            time.sleep(5)
            if tries > 10:
                raise
    return connection


@dataclasses.dataclass
class Shot:
    shot: int = 196560
    channels: Iterable = tuple()
    with_other_signals: bool = False
    only_8x8: bool = False
    only_standard_8x8: bool = False
    only_pos_ip: bool = False
    only_neg_bt: bool = False
    max_delz: float = None
    min_pinj_15l: float = None
    connection: MDSplus.Connection = None
    quiet: bool = False

    def __post_init__(self):
        t1 = time.time()
        if self.connection is None:
            self.connection = make_mdsplus_connection()
        self.channels = np.array(self.channels, dtype=int)
        self.signals = None
        self.metadata = None
        self.is_8x8 = self.is_standard_8x8 = False
        self.only_8x8 = self.only_8x8 or self.only_standard_8x8
        # get BES time array
        self.time = np.array(self.connection.get(f'dim_of(ptdata("besfu01", {self.shot}))'))
        assert self.time.size > 0
        # get BES configuration metadata
        self.connection.openTree('bes', self.shot)
        r_position = np.array(self.connection.get(r'\bes_r'), dtype=np.float32)
        z_position = -np.array(self.connection.get(r'\bes_z'), dtype=np.float32)
        assert r_position.size == 64 and z_position.size == 64
        self.connection.closeTree('bes', self.shot)
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
            inboard_column_channel_order = np.flip(z_first_column.argsort()) * 8
            self.is_standard_8x8 = np.array_equal(
                inboard_column_channel_order,
                np.arange(8, dtype=int) * 8,
            )
        if self.only_8x8: assert self.is_8x8
        if self.only_standard_8x8: assert self.is_standard_8x8
        delz_avg = np.abs(np.diff(z_position[np.arange(8)*8]).mean()).round(2)
        if self.max_delz: assert delz_avg <= self.max_delz
        # metadata
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
            'inboard_column_channel_order': inboard_column_channel_order if self.is_8x8 else None,
            'delz_avg': delz_avg if self.is_8x8 else None,
        }
        # get ip, beams, etc.
        for node_name in ['ip', 'bt']:
            result = self.get_signal(node_name, max_sample_rate=5e3)
            signal:np.ndarray = result['data']
            self.metadata[node_name] = (
                signal.max()
                if np.abs(signal.max()) > np.abs(signal.min())
                else signal.min()
            )
            if self.only_pos_ip and node_name=='ip': assert self.metadata[node_name] > 0
            if self.only_neg_bt and node_name=='bt': assert self.metadata[node_name] < 0
            if self.with_other_signals:
                setattr(self, node_name, result['data'])
                setattr(self, f'{node_name}_time', result['time'])
        for node_name in ['pinj', 'pinj_15l', 'pinj_15r']:
            result = self.get_signal(node_name, tree='nb', max_sample_rate=5e3)
            setattr(self, node_name, result['data'])
            self.metadata[node_name] = result['data'].max()
            if self.min_pinj_15l and node_name=='pinj_15l':  assert self.metadata[node_name] > self.min_pinj_15l
            if node_name == 'pinj':
                setattr(self, f'{node_name}_time', result['time'])
                self.connection.openTree('nb', self.shot)
                date = self.connection.get(f'getnci(\\pinj, "time_inserted")')
                self.metadata['date'] = str(date.date)
                self.connection.closeTree('nb', self.shot)
        if not self.quiet: print(f'{self.shot}: Metadata time = {time.time() - t1:.2f} s')
        # get BES signals
        if self.channels.size > 0:
            self.get_bes_signals()
        # get other signals
        if self.with_other_signals:
            self.get_other_siganls()

    def get_bes_signals(
        self,
        channels: Iterable = tuple(),
    ):
        self.channels = np.array(channels, dtype=int) if channels else self.channels
        assert np.all(self.channels > 0)
        self.signals = np.empty([self.channels.size, self.time.size])
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
            self.signals[i, :] = np.array(self.connection.get(tdi_var))
        if not self.quiet: print(f'{self.shot}: Signal time = {time.time() - t1:.2f} s')

    def get_other_siganls(
            self,
    ):
        pass

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
                    if max_shots and irow+1 > max_shots: break
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
                    result = self._check_channel_configuration(
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
    
    def filter_shots(
            self,
            r_avg=None,
            z_avg=None,
            ip=None,
            bt=None,
            pinj_15l=None,
            pinj_15r=None,
            only_standard_8x8=True,
            export_csv=True,
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
            quiet=True,
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
            shot_group.attrs['configuration_index'] = config_index
            for attr_name, attr_value in bes_data.metadata.items():
                shot_group.attrs[attr_name] = attr_value
            for point_name in ['pinj', 'pinj_15l', 'pinj_15r']:
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
    # bes_data = Shot()
    # bes_data.get_bes_signals([1,2])

    dataset = HDF5_Data()
    dataset.load_shotlist(truncate_hdf5=True)
    dataset.print_hdf5_contents()

    # dataset = HDF5_Data(
    #     # hdf5_file='/home/smithdr/ml/elm_data/step_2_shot_metadata/metadata_v3.hdf5',
    #     hdf5_file='/home/smithdr/ml/elm_data/step_2_shot_metadata/metadata_v5.hdf5',
    # )
    # dataset.print_hdf5_contents()
