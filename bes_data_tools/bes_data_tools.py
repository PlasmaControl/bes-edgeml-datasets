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


class BES_Shot_Data:
    _point_names = [
        'ip',
        'bt',
        'pinj',
        'pinj_15l',
        'pinj_15r',
    ]

    def __init__(
        self,
        shot=196491,
        channels=None,
        only_8x8=False,
        only_standard_8x8=False,
        connection=None,
    ):
        t1 = time.time()
        self.connection = connection
        if self.connection is None:
            tries = 0
            while True:
                tries += 1
                try:
                    self.connection = MDSplus.Connection('atlas.gat.com')
                except:
                    time.sleep(5)
                    if tries <= 10:
                        continue
                    else:
                        raise
                break
        if channels is None:
            channels = np.arange(1, 65)
        channels = np.array(channels, dtype=int)
        self.shot = shot
        self.channels = channels
        self.time = None
        self.signals = None
        self.metadata = None
        self.inboard_column_channel_order = None
        self.is_8x8 = self.is_standard_8x8 = False
        only_8x8 = only_8x8 or only_standard_8x8
        # get time array
        try:
            ptdata = f'ptdata("besfu01", {self.shot})'
            sigtime = self.connection.get(f'dim_of({ptdata})')
            self.time = np.array(sigtime).round(4)
            # get metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r')).round(1)
            z_position = -np.array(self.connection.get(r'\bes_z')).round(1)
            assert r_position.size == 64 and z_position.size == 64
            self.connection.closeTree('bes', self.shot)
        except:
            self.time = None
            print(f'{self.shot}: error with BES data')
            return
        for i in np.arange(8):
            # del-r of column i
            rdiff = np.diff(r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.12)
            # del-z of row i
            zdiff = np.diff(z_position[i * 8 + np.arange(8)])
            row_test = np.all(np.abs(zdiff) <= 0.12)
            self.is_8x8 = col_test and row_test
        if self.is_8x8:
            z_first_column = z_position[np.arange(8) * 8]
            self.inboard_column_channel_order = np.flip(z_first_column.argsort()) * 8
            self.is_standard_8x8 = np.array_equal(
                self.inboard_column_channel_order,
                np.arange(8, dtype=int) * 8,
            )
        if only_8x8 and not self.is_8x8:
            self.time = None
            print(f'{self.shot}: ERROR not 8x8 config')
            return
        if only_standard_8x8 and not self.is_standard_8x8:
            print(f'{self.shot}: ERROR not standard 8x8 config')
            self.time = None
            return
        self.metadata = {
            'shot': self.shot,
            'delta_time': np.diff(self.time[0:100]).mean().round(4),
            'start_time': self.time[0],
            'stop_time': self.time[-1],
            'n_time': self.time.size,
            'time_units': 'ms',
            'r_position': r_position,
            'z_position': z_position,
            'rz_units': 'cm',
            'date': '',
            'ip': 0.,
            'bt': 0.,
            'is_8x8': self.is_8x8,
            'is_standard_8x8': self.is_standard_8x8,
            'r_avg': np.mean(r_position).round(1) if self.is_8x8 else 0.,
            'z_avg': np.mean(z_position).round(1) if self.is_8x8 else 0.,
            'inboard_column_channel_order': self.inboard_column_channel_order,
        }
        # get ip, beams, etc.
        for point_name in self._point_names:
            try:
                if point_name.startswith('pinj'):
                    self.connection.openTree('nb', self.shot)
                    data = np.array(self.connection.get(f'\\{point_name}'), dtype=np.float32)[::4]
                    data_time = np.array(
                            self.connection.get(f'dim_of(\\{point_name})'), dtype=np.float32)[::4]
                    if point_name == 'pinj':
                        date = self.connection.get(
                            f'getnci(\\{point_name}, "time_inserted")')
                        self.metadata['date'] = str(date.date)
                    self.connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(self.connection.get(ptdata), dtype=np.float32)[::8]
                    data_time = np.array(self.connection.get('dim_of(_n)'), dtype=np.float32)[::8]
                time_mask = np.logical_and(data_time >= self.time[0],
                                           data_time <= self.time[-1])
                data = data[time_mask]
                data_time = data_time[time_mask]
            except:
                self.time = None
                print(f'{self.shot}: ERROR for data node {point_name}')
                return
            assert data.shape == data_time.shape
            setattr(self, point_name, data)
            if point_name == 'pinj' or 'pinj' not in point_name:
                setattr(self, f'{point_name}_time', data_time)
        pinj_15l_max = np.max(getattr(self, 'pinj_15l'))
        if pinj_15l_max < 500e3:
            self.time = None
            print(f'{self.shot}: ERROR small pinj_15l')
            return
        for ptname in ['ip', 'bt']:
            try:
                data: np.ndarray = getattr(self, ptname)
                if np.abs(data.max()) >= np.abs(data.min()):
                    self.metadata[ptname] = data.max()
                else:
                    self.metadata[ptname] = data.min()
            except:
                self.time = None
                print(f'{self.shot}: ERROR {ptname}')
                return
        print(f'{self.shot}: {self.time.size} time points')
        print(f'{self.shot}: Metadata time = {time.time() - t1:.2f} s')

    def get_signals(
        self,
        channels=None,
    ):
        assert self.time is not None
        if channels:
            self.channels = np.array(channels, dtype=int)
        t1 = time.time()
        print(f'{self.shot}: fetching {self.channels.size} signals')
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


class BES_Metadata:

    def __init__(
        self,
        hdf5_file: str|Path = 'bes_data.hdf5',
    ) -> None:
        self.hdf5_file = Path(hdf5_file)

    def load_shotlist(
        self,
        csv_file: str|Path = '',
        shotlist: Iterable[int] = (196554,196555,196559,196560),
        truncate_hdf5: bool = False,
        max_shots: int = None,
        use_concurrent: bool = False,
        max_workers: int = None,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
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
                                    valid_shot_count += 0
                        with open('futures_status.txt', 'w') as txt_file:
                            txt_file.write(f"{datetime.now()}\n")
                            txt_file.write(f"Total futures: {len(futures)}\n")
                            txt_file.write(f"Running: {running}\n")
                            txt_file.write(f"Done: {done}\n")
                            txt_file.write(f"Valid shots: {valid_shot_count}\n")
                        if running == 0:
                            break
                    # shot_count = 0
                    # for future in concurrent.futures.as_completed(futures):
                    #     shot_count += 1
                    #     result = future.result()
                    #     if future.exception() is None and result:
                    #         print(f'Shot {result} is good ({shot_count} of {shotlist.size})')
                    #         valid_shot_count += 1
                    #     else:
                    #         print(f'Shot {result} is bad ({shot_count} of {shotlist.size})')
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
        lock: threading.Lock = None,
        connection = None,
    ) -> int:
        bes_data = BES_Shot_Data(
            shot=shot,
            only_8x8=only_8x8,
            only_standard_8x8=only_standard_8x8,
            connection=connection,
        )
        if bes_data.time is None:
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
                    new_config.attrs['r_avg'] = bes_data.metadata['r_avg']
                    new_config.attrs['z_avg'] = bes_data.metadata['z_avg']
                    new_config.attrs['inboard_column_channel_order'] = bes_data.inboard_column_channel_order
                else:
                    config_index = config_index_count[1] + 1
                    new_config = group_non_8x8.create_group(f'{config_index:04d}')
                new_config.attrs['r_position'] = bes_data.metadata['r_position']
                new_config.attrs['z_position'] = bes_data.metadata['z_position']
                new_config.attrs['is_8x8'] = bes_data.is_8x8
                new_config.attrs['is_standard_8x8'] = bes_data.is_standard_8x8
                new_config.attrs['shots'] = np.array([bes_data.shot], dtype=int)
                new_config.attrs['n_shots'] = new_config.attrs['shots'].size
            assert config_index
            # save shot metadata
            shot_group = h5root.require_group(str(shot))
            for attr_name, attr_value in bes_data.metadata.items():
                shot_group.attrs[attr_name] = attr_value
                shot_group.attrs['configuration_index'] = config_index
            for point_name in bes_data._point_names:
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
        return shot
    
    def print_hdf5_contents(self) -> None:
        def print_attributes(obj) -> None:
            for key, value in obj.attrs.items():
                if isinstance(value, np.ndarray):
                    print(f'  Attribute {key}:', value.shape, value.dtype)
                else:
                    print(f'  Attribute {key}:', value)
        def recursively_print_content(group: h5py.Group) -> None:
            print(f'Group {group.name}')
            print_attributes(group)
            for key, value in group.items():
                if isinstance(value, h5py.Group):
                    recursively_print_content(value)
                if isinstance(value, h5py.Dataset):
                    print(f'  Dataset {key}:', value.shape, value.dtype)
                    print_attributes(value)
        print(f'Contents of {self.hdf5_file}')
        assert self.hdf5_file.exists()
        with h5py.File(self.hdf5_file, 'r') as h5file:
            recursively_print_content(h5file)

    def print_hdf5_summary(self) -> None:
        assert self.hdf5_file.exists()

    def plot_8x8_configurations(
        self,
    ) -> None:
        assert self.hdf5_file.exists()
        with h5py.File(self.hdf5_file, 'r') as hfile:
            total_shots = 0
            plt.figure(figsize=(4,4))
            plt.subplot(111)
            for shot_key, shot_group in hfile.items():
                if shot_key.startswith('config'):
                    continue
                plt.plot(shot_group.attrs['r_avg'], shot_group.attrs['z_avg'], 
                         marker='x', ms=4, color='C0', alpha=0.5)
            plt.xlim(218, 232)
            plt.xlabel('R (cm)')
            plt.ylim(-8, 8)
            plt.ylabel('Z (cm)')
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig('8x8_configurations.pdf', format='pdf')

    def save_signals(
        self,
    ) -> None:
        assert self.hdf5_file.exists()
        

if __name__=='__main__':
    # bes_data = BES_Shot_Data()
    # bes_data.get_signals([1,2])

    dataset = BES_Metadata()
    dataset.load_shotlist(truncate_hdf5=True, use_concurrent=True, max_workers=2)
    dataset.print_hdf5_contents()
    dataset.plot_8x8_configurations()