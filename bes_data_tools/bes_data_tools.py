import time
from typing import Iterable
import os
import csv
import threading
import concurrent.futures
import contextlib
from pathlib import Path

import numpy as np
import MDSplus
import h5py


class BES_Shot_Data:
    _points = [
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
    ):
        t1 = time.time()
        self.connection = MDSplus.Connection('atlas.gat.com')
        if channels is None:
            channels = np.arange(1, 65)
        channels = np.array(channels, dtype=int)
        self.shot = shot
        self.channels = channels
        self.time = None
        self.signals = None
        self.metadata = None
        only_8x8 = only_8x8 or only_standard_8x8
        # get time array
        try:
            ptdata = f'ptdata("besfu01", {self.shot})'
            sigtime = self.connection.get(f'dim_of({ptdata})')
            self.time = np.array(sigtime).round(4)
            # get metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r')).round(2)
            z_position = -np.array(self.connection.get(r'\bes_z')).round(2)
            assert r_position.size == 64 and z_position.size == 64
            self.connection.closeTree('bes', self.shot)
        except:
            self.time = None
            print(f'{self.shot}: error with BES data')
            return
        self.is_8x8 = self.is_standard_8x8 = False
        for i in np.arange(8):
            # del-r of column i
            rdiff = np.diff(r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.13)
            # del-z of row i
            zdiff = np.diff(z_position[i * 8 + np.arange(8)])
            row_test = np.all(np.abs(zdiff) <= 0.13)
            self.is_8x8 = col_test and row_test
        if self.is_8x8:
            z_first_column = z_position[np.arange(8) * 8]
            self.is_standard_8x8 = (z_first_column.argmax() == 0) and \
                                   (z_first_column.argmin() * 8 == 56)
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
        }
        # get ip, beams, etc.
        for point_name in self._points:
            try:
                if point_name.startswith('pinj'):
                    self.connection.openTree('nb', self.shot)
                    data = np.array(self.connection.get(f'\\{point_name}'))[::2]
                    data_time = np.array(
                            self.connection.get(f'dim_of(\\{point_name})'))[::2]
                    if point_name == 'pinj':
                        date = self.connection.get(
                            f'getnci(\\{point_name}, "time_inserted")')
                        self.metadata['date'] = str(date.date)
                    self.connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(self.connection.get(ptdata))[::8]
                    data_time = np.array(self.connection.get('dim_of(_n)'))[::8]
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


class BES_HDF5_Dataset:

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
        if truncate_hdf5:
            self.hdf5_file.unlink(missing_ok=True)
        only_8x8 = only_8x8 and only_standard_8x8
        with h5py.File(self.hdf5_file, 'a') as h5root:
            valid_shot_counter = 0
            configuration_group = h5root.require_group('configurations')
            group_8x8 = configuration_group.require_group('8x8_configurations')
            group_non_8x8 = configuration_group.require_group('non_8x8_configurations')
            if use_concurrent:
                pass
            else:
                for i, shot in enumerate(shotlist):
                    print(f'Shot {shot} ({i + 1} of {shotlist.size})')
                    result = self.validate_bes_shot(
                        shot=shot,
                        group_8x8=group_8x8,
                        group_non_8x8=group_non_8x8,
                        only_8x8=only_8x8,
                        only_standard_8x8=only_standard_8x8,
                    )
                    if result:
                        valid_shot_counter += 1
    
    def validate_bes_shot(
        self,
        shot: int,
        group_8x8: h5py.Group,
        group_non_8x8: h5py.Group,
        only_8x8: bool = False,
        only_standard_8x8: bool = False,
        lock: threading.Lock = None,
    ) -> bool:
        bes_shot = BES_Shot_Data(
            shot=shot,
            only_8x8=only_8x8,
            only_standard_8x8=only_standard_8x8,
        )
        if bes_shot.time is None:
            return False
        if lock is None:
            lock = contextlib.nullcontext()
        with lock:
            config_index = None
            for config_group in [group_8x8, group_non_8x8]:
                for config_index_str, config in config_group.items():
                    is_same = False
                    for position in ['r_position','z_position']:
                        is_same = np.allclose(
                            bes_data.metadata[position],
                            config.attrs[position],
                            atol=0.12,
                        )
                        if not is_same:
                            break
                    if is_same:
                        if shot not in config.attrs['shots']:
                            config.attrs['shots'] = np.append(config.attrs['shots'], shot)
                            config.attrs['nshots'] = config.attrs['shots'].size
                        config_index = int(config_index_str)
                    if config_index:
                        break
                if config_index:
                    break
            if config_index is None:
                pass
        return True
        



if __name__=='__main__':
    bes_data = BES_Shot_Data()
    # bes_data.get_signals([1,2])

    dataset = BES_HDF5_Dataset()
    dataset.load_shotlist()