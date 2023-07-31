import time
import os
import csv
import threading
import concurrent.futures
from pathlib import Path

import numpy as np
import MDSplus
import h5py


class BES_Data(object):
    _points = ['ip',
               'bt',
               'pinj',
               'pinj_15l',
               'pinj_15r',
               ]

    def __init__(
            self,
            shot=176778,
            channels=None,
            only_8x8=True,
            only_standard_8x8=True,
        ):
        t1 = time.time()
        self.connection = MDSplus.Connection('atlas.gat.com')
        if channels is None:
            channels = np.arange(1, 65)
        channels = np.array(channels, dtype=int)
        self.shot = shot
        self.channels = channels
        self.time = None
        self.n_time = None
        self.signals = None
        self.metadata = None
        only_8x8 = only_8x8 or only_standard_8x8
        # get time array
        ptdata = f'ptdata("bessu01", {self.shot})'
        try:
            sigtime = self.connection.get(f'dim_of({ptdata})')
            np.array(sigtime).round(4)
        except:
            print(f'{self.shot}: ERROR no BES slow time data')
            self.time = None
            return
        ptdata = f'ptdata("besfu01", {self.shot})'
        try:
            sigtime = self.connection.get(f'dim_of({ptdata})')
            self.time = np.array(sigtime).round(4)
        except:
            self.time = None
            print(f'{self.shot}: ERROR no BES fast time data')
            return
        n_time = self.connection.get(f'size({ptdata})')
        self.n_time = n_time.data()
        assert (self.n_time == self.time.size)
        try:
            # get metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r')).round(2)
            z_position = -np.array(self.connection.get(r'\bes_z')).round(2)
            assert r_position.size == 64 and z_position.size == 64
            self.connection.closeTree('bes', self.shot)
        except:
            print(f'{self.shot}: ERROR getting BES position metadata')
            self.time = None
            return
        is_8x8 = is_standard_8x8 = False
        for i in np.arange(8):
            # del-r of column i
            rdiff = np.diff(r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.13)
            # del-z of row i
            zdiff = np.diff(z_position[i * 8 + np.arange(8)])
            row_test = np.all(np.abs(zdiff) <= 0.13)
            is_8x8 = col_test and row_test
        if is_8x8:
            z_first_column = z_position[np.arange(8) * 8]
            is_standard_8x8 = (z_first_column.argmax() * 8 == 0) and \
                              (z_first_column.argmin() * 8 == 56)
        if only_8x8 and not is_8x8:
            self.time = None
            print(f'{self.shot}: ERROR not 8x8 config')
            return
        if only_standard_8x8 and not is_standard_8x8:
            print(f'{self.shot}: ERROR not standard 8x8 config')
            self.time = None
            return
        self.metadata = {
            'shot': self.shot,
            'delta_time': np.diff(self.time[0:100]).mean().round(4),
            'start_time': self.time[0],
            'stop_time': self.time[-1],
            'n_time': self.n_time,
            'time_units': 'ms',
            'r_position': r_position,
            'z_position': z_position,
            'rz_units': 'cm',
            'date': '',
            'ip': 0.,
            'bt': 0.,
            'is_8x8': is_8x8,
            'is_standard_8x8': is_standard_8x8,
            'r_avg': np.mean(r_position).round(1) if is_8x8 else 0.,
            'z_avg': np.mean(z_position).round(1) if is_8x8 else 0.,
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
                        self.metadata['date'] = date.date.decode('utf-8')
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
        if self.pinj_15l.max() < 500e3:
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
        print(f'{self.shot}: {self.n_time} time points')
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
        self.signals = np.empty([self.channels.size, self.n_time])
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


def _validate_configuration(input_bes_data,
                            config_8x8_group,
                            config_non_8x8_group):
    max_index = np.array([0, 100])
    r_position = input_bes_data.metadata['r_position']
    z_position = input_bes_data.metadata['z_position']
    for igroup, config_group in enumerate([config_8x8_group,
                                           config_non_8x8_group]):
        for config_index_str, config in config_group.items():
            if config_index_str.startswith('0'):
                config_index_str = config_index_str[1]
            config_index = eval(config_index_str)
            assert (isinstance(config, h5py.Group))
            assert ('r_position' in config.attrs and
                    'z_position' in config.attrs and
                    'shots' in config.attrs)
            max_index[igroup] = np.max([max_index[igroup],
                                        config_index])
            # test if input data matches existing configuration
            if not np.allclose(r_position,
                               config.attrs['r_position'],
                               atol=0.1):
                continue
            if not np.allclose(z_position,
                               config.attrs['z_position'],
                               atol=0.1):
                continue
            print(f'{input_bes_data.shot}: Configuration matches index {config_index}')
            if input_bes_data.shot not in config.attrs['shots']:
                config.attrs['shots'] = np.append(config.attrs['shots'],
                                                  input_bes_data.shot)
                config.attrs['nshots'] = config.attrs['shots'].size
            return config_index
    print(f'{input_bes_data.shot}: Configuration does not match existing configuration')
    # now test for 8x8 configuration
    config_is_8x8 = True
    for i in np.arange(8):
        rdiff = np.diff(r_position[i + np.arange(8) * 8])
        col_test = np.allclose(rdiff, np.zeros(rdiff.shape), atol=0.1)
        zdiff = np.diff(z_position[i * 8 + np.arange(8)])
        row_test = np.allclose(zdiff, np.zeros(zdiff.shape), atol=0.1)
        config_is_8x8 = config_is_8x8 and col_test and row_test
        if not config_is_8x8:
            break
    if config_is_8x8:
        new_index = max_index[0] + 1
        print(f'{input_bes_data.shot}: New 8x8 config index is {new_index}')
        new_config = config_8x8_group.create_group(f'{new_index:02d}')
        new_config.attrs['r_avg'] = np.mean(r_position).round(2)
        new_config.attrs['z_avg'] = np.mean(z_position).round(2)
        z_first_column = z_position[np.arange(8) * 8]
        new_config.attrs['upper_inboard_channel'] = z_first_column.argmax() * 8
        new_config.attrs['lower_inboard_channel'] = z_first_column.argmin() * 8
    else:
        new_index = max_index[1] + 1
        print(f'{input_bes_data.shot}: New non-8x8 config index is {new_index}')
        new_config = config_non_8x8_group.create_group(f'{new_index:d}')
    new_config.attrs['r_position'] = r_position
    new_config.attrs['z_position'] = z_position
    new_config.attrs['shots'] = np.array([input_bes_data.shot], dtype=np.int)
    new_config.attrs['nshots'] = new_config.attrs['shots'].size
    return new_index


def validate_bes_shot(shot=None,
                      with_signals=False,
                      metafile=None,
                      lock=None,
                      only_8x8=False,
                      ):

    bes_data = BES_Data(shot=shot)
    if bes_data.time is None:
        print(f'{bes_data.shot}: ERROR invalid BES_Data object')
        return -shot
    # metadata attributes
    if lock:
        lock.acquire()
    configuration_group = metafile.require_group('configurations')
    config_8x8_group = configuration_group.require_group('8x8_configurations')
    config_non_8x8_group = configuration_group.require_group('non_8x8_configurations')
    shot_group = metafile.require_group(f'{bes_data.shot}')
    for attr_name, attr_value in bes_data.metadata.items():
        if attr_name in shot_group.attrs:
            if 'position' in attr_name:
                assert (np.allclose(attr_value,
                                    shot_group.attrs[attr_name],
                                    atol=0.1))
            else:
                assert (attr_value == shot_group.attrs[attr_name])
        else:
            shot_group.attrs[attr_name] = attr_value
    config_index = _validate_configuration(bes_data,
                                           config_8x8_group,
                                           config_non_8x8_group)
    if 'configuration_index' in shot_group.attrs:
        assert (config_index == shot_group.attrs['configuration_index'])
    else:
        shot_group.attrs['configuration_index'] = config_index
    # metadata datasets
    for point_name in bes_data._points:
        for name in [f'{point_name}', f'{point_name}_time']:
            data = getattr(bes_data, name, None)
            if data is None:
                continue
            shot_group.require_dataset(name,
                                       data=data,
                                       shape=data.shape,
                                       dtype=data.dtype)
    if lock:
        lock.release()
    # signals
    if with_signals:
        signals_dir = Path('signals')
        signals_dir.mkdir(exist_ok=True)
        if bes_data.signals is None:
            print(f'{bes_data.shot}: ERROR invalid BES signals')
            return -bes_data.shot
        bes_data.get_signals()
        signal_file = signals_dir / f'bes_signals_{bes_data.shot}.hdf5'
        with h5py.File(signal_file.as_posix(), 'w') as sfile:
            sfile.create_dataset('signals',
                                 data=bes_data.signals,
                                 compression='gzip',
                                 chunks=True)
            sfile.create_dataset('time',
                                 data=bes_data.time,
                                 compression='gzip',
                                 chunks=True)
        # print_h5py_contents(signal_file)
        # signal_mb = bes_data.signals.nbytes // 1024 // 1024
        # print(f'{bes_data.shot}: BES_Data size = {signal_mb} MB')
    return shot


def package_bes_data(
    shotlist=(176778, 171472, 184573, 178556),
    input_shotlist_csv=None,
    max_shots=None,
    output_hdf5='metadata.hdf5',
    with_signals=False,
    use_concurrent=False,
    max_workers=None,
):
    t1 = time.time()
    if input_shotlist_csv:
        input_shotlist_csv = Path(input_shotlist_csv)
        assert input_shotlist_csv.exists()
        print(f'Shotlist: {input_shotlist_csv.as_posix()}')
        shotlist = []
        with input_shotlist_csv.open() as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            assert reader.fieldnames[0] == 'shot'
            for irow, row in enumerate(reader):
                if max_shots and irow+1 > max_shots:
                    break
                shotlist.append(int(row['shot']))
    shotlist = np.array(shotlist, dtype=int)
    output_hdf5 = Path(output_hdf5)
    with h5py.File(output_hdf5.as_posix(), 'w') as metadata_file:
        valid_shot_counter = 0
        if use_concurrent:
            if not max_workers:
                max_workers = len(os.sched_getaffinity(0)) // 2
            lock = threading.Lock()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                # submit tasks to workers
                for i, shot in enumerate(shotlist):
                    print(f'{shot}: submitting to worker pool ({i+1} of {shotlist.size})')
                    future = executor.submit(
                        validate_bes_shot,
                        shot=shot,
                        with_signals=with_signals,
                        metafile=metadata_file,
                        lock=lock,
                    )
                    futures.append(future)
                # get results as workers finish
                shot_count = 0
                for future in concurrent.futures.as_completed(futures):
                    shot_count += 1
                    shot = future.result()
                    if future.exception() is None and shot > 0:
                        valid_shot_counter += 1
                        print(f'{shot}: work finished ({shot_count} of {shotlist.size})')
                    else:
                        print(f'{-shot}: INVALID return value')
                t_mid = time.time()
                print(f'Worker pool elapsed time = {t_mid-t1:.2f} s')
        else:
            for i, shot in enumerate(shotlist):
                print(f'Trying {shot} ({i + 1} of {shotlist.size})')
                shot = validate_bes_shot(
                    shot=shot,
                    with_signals=with_signals,
                    metafile=metadata_file,
                )
                if shot and shot>0:
                    valid_shot_counter += 1
                    print( f'{shot} good')
                else:
                    print(f'{-shot} INVALID return value')
    t2 = time.time()
    print_metadata_contents(input_hdf5file=output_hdf5)
    dt = t2 - t1
    print(f'Packaging data elapsed time: {int(dt)//3600} hr {dt%3600/60:.1f} min')
    print(f'{valid_shot_counter} valid shots out of {shotlist.size} in input shot list')



if __name__=='__main__':
    bes_data = BES_Data(shot=184800)
    bes_data.get_signals([1,2])