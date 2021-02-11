from pathlib import Path
import time
import numpy as np
import h5py
import MDSplus
import matplotlib.pyplot as plt
import concurrent.futures
import os
import sys


def _print_attrs(obj):
    for attr_name, attr_value in obj.attrs.items():
        if isinstance(attr_value, np.ndarray) and attr_value.size > 4:
            if np.issubdtype(attr_value.dtype, np.floating):
                tmp = [f'{val:.2f}' for val in attr_value[0:4]]
            else:
                tmp = [f'{val}' for val in attr_value[0:4]]
            print_value = '[ ' + ' '.join(tmp) + ' ... ]'
        else:
            if hasattr(attr_value, 'dtype') and np.issubdtype(attr_value.dtype,
                                                              np.floating):
                print_value = f'{attr_value:.2f}'
            else:
                print_value = attr_value
        print(f'    Attribute: {attr_name} {print_value}')


def traverse_h5py(group):
    """
    Recursively traverse hdf5 file or group, and print summary information
    on subgroups, datasets, and attributes
    """
    do_close = False
    # open h5 file if `group` is path
    if isinstance(group, (str, Path)):
        do_close = True
        if isinstance(group, str):
            group = h5py.File(group, 'r')
        else:
            group = h5py.File(group.as_posix(), 'r')
    print(f'Group {group.name} in file {group.file}')
    _print_attrs(group)
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            traverse_h5py(value)
        if isinstance(value, h5py.Dataset):
            print(f'    Dataset {value.name}', value.shape, value.dtype)
            _print_attrs(value)
    if do_close:
        group.close()


class BES_Data(object):
    _points = ['ip',
               'bt',
               'pinj',
               'pinj_15l',
               'vinj_15l',
               'pinj_15r',
               'vinj_15r',
               ]

    def __init__(self,
                 shot=None,
                 channels=None,
                 verbose=False,
                 get_signals=False):
        t1 = time.time()
        self.connection = MDSplus.Connection('atlas.gat.com')
        if shot is None:
            shot = 176778
        if channels is None:
            channels = np.arange(1, 65)
        channels = np.array(channels)
        self.shot = shot
        self.channels = channels
        self.verbose = verbose
        self.signals = None
        self.metadata = None
        print(f'{self.shot}: start')
        # get time array
        ptdata = f'ptdata("besfu01", {self.shot})'
        try:
            self.time = np.array(self.connection.get(f'dim_of({ptdata})')).round(4)
        except:
            self.time = None
            return
        n_time = self.connection.get(f'size({ptdata})')
        self.n_time = n_time.data()
        assert (self.n_time == self.time.size)
        try:
            # get metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r')).round(2)
            z_position = np.array(self.connection.get(r'\bes_z')).round(2)
            start_time = self.connection.get(r'\bes_ts')
            self.connection.closeTree('bes', self.shot)
        except:
            self.time = None
            return
        if not start_time == self.time[0]:
            print(f'{self.shot}: ALERT inconsistent start times: ',
                  start_time, self.time[0])
        self.metadata = {'shot': self.shot,
                         'delta_time': np.diff(self.time[0:100]).mean().round(
                             4),
                         'start_time': self.time[0],
                         'stop_time': self.time[-1],
                         'n_time': self.n_time,
                         'time_units': 'ms',
                         'r_position': r_position,
                         'z_position': z_position,
                         'rz_units': 'cm',
                         'date': ''}
        # get ip, beams, etc.
        for point_name in self._points:
            data = np.array(0)
            data_time = np.array(0)
            try:
                if 'inj' in point_name:
                    self.connection.openTree('nb', self.shot)
                    data = np.array(self.connection.get(f'\\{point_name}'))
                    data_time = np.array(
                            self.connection.get(f'dim_of(\\{point_name})'))
                    if point_name == 'pinj':
                        date = self.connection.get(
                            f'getnci(\\{point_name}, "time_inserted")')
                        self.metadata['date'] = date.date.decode('utf-8')
                    self.connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(self.connection.get(ptdata))
                    data_time = np.array(self.connection.get('dim_of(_n)'))
                time_mask = np.logical_and(data_time >= self.time[0],
                                           data_time <= self.time[-1])
                data = data[time_mask]
                data_time = data_time[time_mask]
            except:
                print(f'{self.shot}: INVALID data node for {point_name}')
                data = h5py.Empty(dtype='f')
                data_time = h5py.Empty(dtype='f')
            assert (data.shape == data_time.shape)
            setattr(self, point_name, data)
            if point_name == 'pinj' or 'inj' not in point_name:
                setattr(self, f'{point_name}_time', data_time)
        print(f'{self.shot}: {self.n_time} time points')
        t2 = time.time()
        print(f'{self.shot}: Metadata time = {t2 - t1:.2f} s')
        if get_signals:
            self.get_signals()

    def get_signals(self):
        t1 = time.time()
        print(f'{self.shot}: fetching {self.channels.size} signals')
        tdi_vars = []
        tdi_assignments = []
        for channel in self.channels:
            var = f'_n{channel:02d}_{self.shot}'
            tdi_vars.append(var)
            tmp = f'{var} = ptdata("besfu{channel:02d}", {self.shot})'
            tdi_assignments.append(tmp)
        self.connection.get(', '.join(tdi_assignments))
        self.signals = np.empty([self.channels.size, self.n_time])
        for i, tdi_var in enumerate(tdi_vars):
            self.signals[i, :] = self.connection.get(tdi_var)
        t2 = time.time()
        print(f'{self.shot}: Signal time = {t2 - t1:.2f} s')


def validate_configuration(input_bes_data,
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


def package_bes(filename=None,
                shots=None,
                channels=None,
                verbose=False,
                with_signals=False):
    if filename is None:
        filename = 'bes_metadata.hdf5'
    if shots is None:
        shots = [176778, 171472]
    if channels is None:
        channels = np.arange(1,65)
    shots = np.array(shots)
    channels = np.array(channels)
    t1 = time.time()
    with h5py.File(filename, 'a') as metafile:
        valid_shot_counter = 0
        configuration_group = metafile.require_group('configurations')
        config_8x8_group = configuration_group.require_group(
            '8x8_configurations')
        config_non_8x8_group = configuration_group.require_group(
            'non_8x8_configurations')
        max_workers = len(os.sched_getaffinity(0))//2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            shot_count = 0
            for i, shot in enumerate(shots):
                print(f'{shot}: submitting to worker pool ({i+1} of {shots.size})')
                future = executor.submit(BES_Data,
                                         shot=shot,
                                         channels=channels,
                                         verbose=False,
                                         get_signals=with_signals)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                shot_count += 1
                bes_data = future.result()
                print(f'{bes_data.shot}: work finished ({shot_count} of {shots.size})')
            t_mid = time.time()
            print(f'Worker pool elapsed time = {t_mid-t1:.2f} s')
            for future in futures:
                t3 = time.time()
                bes_data = future.result()
                if bes_data.time is None:
                    print(f'{bes_data.shot}: INVALID BES data')
                    continue
                shot_string = f'{bes_data.shot:d}'
                shot_group = metafile.require_group(shot_string)
                # metadata attributes
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
                config_index = validate_configuration(bes_data,
                                                      config_8x8_group,
                                                      config_non_8x8_group)
                if 'configuration_index' in shot_group.attrs:
                    assert(config_index == shot_group.attrs['configuration_index'])
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
                valid_shot_counter += 1
                t4 = time.time()
                print(f'{bes_data.shot}: Metadata validation time = {t4-t3:.2f} s')
                # signals
                if with_signals:
                    signal_file = f'bes_signals_{shot_string}.hdf5'
                    with h5py.File(signal_file, 'w') as sfile:
                        sfile.create_dataset('signals',
                                             data=bes_data.signals,
                                             compression='gzip',
                                             chunks=True)
                        sfile.create_dataset('time',
                                             data=bes_data.time,
                                             compression='gzip',
                                             chunks=True)
                    t5 = time.time()
                    if verbose:
                        traverse_h5py(signal_file)
                    print(f'{bes_data.shot}: Signal validation time = {t5-t4:.2f} s')
                    print(f'{bes_data.shot}: BES_Data size = {bes_data.signals.nbytes // 1024 // 1024} MB')
    t2 = time.time()
    if verbose:
        print_metadata_summary(path=filename)
    print(f'Packaging data elapsed time = {t2 - t1:.2f} s')
    print(f'{valid_shot_counter} valid shots out of {shots.size} in input shot list')


def print_metadata_summary(path=None, only_8x8=False):
    if not path:
        path = 'bes_metadata.hdf5'
    if not isinstance(path, Path):
        path = Path(path)
    print(f'Summarizing metadata file {path.as_posix()}')
    with h5py.File(path, 'r') as metadata_file:
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        config_non_8x8_group = metadata_file['configurations']['non_8x8_configurations']
        if only_8x8:
            traverse_h5py(config_8x8_group)
        else:
            traverse_h5py(metadata_file)
        for group in [config_8x8_group, config_non_8x8_group]:
            sum_shots = 0
            for config_group in group.values():
                nshots = config_group.attrs['shots'].size
                sum_shots += nshots
                print(f'# of shots in {config_group.name}: {nshots}')
            print(f'  Sum of shots in {group.name} group: {sum_shots}')
            if only_8x8:
                break


def make_8x8_sublist(path=None,
                     upper_inboard_channel=None,
                     verbose=False,
                     noplot=False,
                     rminmax=(223,226.5),
                     zminmax=(-1.5,1)):
    if not path:
        path = 'bes_metadata.hdf5'
    if not isinstance(path, Path):
        path = Path(path)
    r = []
    z = []
    nshots = []
    shotlist = np.array((), dtype=np.int)
    with h5py.File(path, 'r') as metadata_file:
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        for name, config in config_8x8_group.items():
            upper = config.attrs['upper_inboard_channel']
            if upper_inboard_channel is not None and upper != upper_inboard_channel:
                continue
            shots = config.attrs['shots']
            r_avg = config.attrs['r_avg']
            z_avg = config.attrs['z_avg']
            nshots.append(shots.size)
            r.append(r_avg)
            z.append(z_avg)
            if rminmax[0] <= r_avg <= rminmax[1] and  zminmax[0] <= z_avg <= zminmax[1]:
                shotlist = np.append(shotlist, shots)
            if verbose:
                print(f'8x8 config #{name} nshots {nshots[-1]} ravg {r_avg:.2f} upper {upper}')
    print(f'Shots within r/z min/max limits: {shotlist.size}')
    if not noplot:
        plt.plot(r, z, 'x')
        for i, nshot in enumerate(nshots):
            plt.annotate(repr(nshot),
                         (r[i], z[i]),
                         textcoords='offset points',
                         xytext=(0,10),
                         ha='center')
        plt.xlim(222, 227)
        plt.ylim(None, 1.5)
        for r in rminmax:
            plt.vlines(r, zminmax[0], zminmax[1], color='k')
        for z in zminmax:
            plt.hlines(z, rminmax[0], rminmax[1], color='k')
        plt.xlabel('R (cm)')
        plt.ylabel('Z (cm)')
        plt.title('R/Z centers of BES 8x8 grids, and shot counts')
    return shotlist


if __name__ == '__main__':
    shotlist = [176778, 171472, 171473, 171477,
                171495, 145747, 145745, 142300,
                142294, 145384, 164895, 164824]
    package_bes(shots=shotlist, channels=np.arange(1,11), verbose=True, with_signals=True)
