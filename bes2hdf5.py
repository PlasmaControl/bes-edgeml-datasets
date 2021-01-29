from pathlib import Path
import time
import pickle
import numpy as np
import h5py
import MDSplus

connection = MDSplus.Connection('atlas.gat.com')

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)


def traverse_h5py(group, verbose=False):
    """
    Recursively traverse hdf5 file or group, and print summary information
    on subgroups, datasets, and attributes
    """
    def print_attrs(obj):
        for attr_name, attr_value in obj.attrs.items():
            if not verbose and isinstance(attr_value, np.ndarray) and attr_value.size>3:
                print_value = attr_value[0:3]
            else:
                print_value = attr_value
            print(f'    Attribute: {attr_name} {print_value}')

    do_close = False
    if isinstance(group, (str, Path)):
        do_close = True
        if isinstance(group, str):
            group = h5py.File(group, 'r')
        else:
            group = h5py.File(group.as_posix(), 'r')
    print(f'Group {group.name} in file {group.file}')
    print_attrs(group)
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            traverse_h5py(value)
        if isinstance(value, h5py.Dataset):
            print(f'    Dataset {value.name}', value.shape, value.dtype)
            print_attrs(value)
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

    def __init__(self, shot=None, channels=None, verbose=False):
        if not shot:
            shot = 176778
        if not channels:
            channels = np.arange(1, 65)
        if not isinstance(channels, np.ndarray):
            channels = np.array(channels)
        self.shot = shot
        self.channels = channels
        self.signals = None
        self.date = None
        self.verbose = verbose
        if self.verbose:
            print(f'Getting time and metadata for shot {self.shot}')
            t1 = time.time()
        # get time array
        ptdata = f'ptdata("besfu01", {self.shot})'
        try:
            self.time = np.array(connection.get(f'dim_of({ptdata})')).round(4)
        except:
            self.time = None
            return
        n_time = connection.get(f'size({ptdata})')
        self.n_time = n_time.data()
        assert (self.n_time == self.time.size)
        try:
            # get metadata
            connection.openTree('bes', self.shot)
            r_position = np.array(connection.get(r'\bes_r')).round(2)
            z_position = np.array(connection.get(r'\bes_z')).round(2)
            start_time = connection.get(r'\bes_ts')
            connection.closeTree('bes', self.shot)
        except:
            self.time = None
            return
        if not start_time == self.time[0]:
            print('ALERT: shot {shot} with inconsistent start times: ',
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
                    connection.openTree('nb', self.shot)
                    data = np.array(connection.get(f'\\{point_name}'))
                    data_time = np.array(
                            connection.get(f'dim_of(\\{point_name})'))
                    if point_name == 'pinj':
                        date = connection.get(
                            f'getnci(\\{point_name}, "time_inserted")')
                        self.metadata['date'] = date.date.decode('utf-8')
                    connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(connection.get(ptdata))
                    data_time = np.array(connection.get('dim_of(_n)'))
                time_mask = np.logical_and(data_time >= self.time[0],
                                           data_time <= self.time[-1])
                data = data[time_mask]
                data_time = data_time[time_mask]
            except:
                print(f'INVALID data node for shot {self.shot}: {point_name}')
                data = h5py.Empty(dtype='f')
                data_time = h5py.Empty(dtype='f')
            assert (data.shape == data_time.shape)
            setattr(self, point_name, data)
            if point_name == 'pinj' or 'inj' not in point_name:
                setattr(self, f'{point_name}_time', data_time)
        if self.verbose:
            t2 = time.time()
            print(f'  Shot {self.shot} with {self.n_time} time points')
            print(f'  Time, metadata elapsed time = {t2 - t1:.2f} s')

    def get_signals(self):
        tdi_vars = []
        tdi_assignments = []
        for channel in self.channels:
            tdi_vars.append(f'_n{channel:02d}')
            tdi_assignments.append(
                    f'{tdi_vars[-1]} = ptdata("besfu{channel:02d}", {self.shot})')
        if self.verbose:
            print(
                    f'  Fetching signals ({self.channels.size} channels) for shot {self.shot}')
            t1 = time.time()
        connection.get(', '.join(tdi_assignments))
        self.signals = np.empty([self.channels.size, self.n_time])
        for i, tdi_var in enumerate(tdi_vars):
            self.signals[i, :] = connection.get(tdi_var)
        if self.verbose:
            t2 = time.time()
            print(f'  Get signals elapsed time = {t2 - t1:.2f} s')


def package_bes(shots=None,
                channels=None,
                verbose=False,
                with_signals=False):
    if shots is None and channels is None:
        shots = [176778, 171472]
        channels = [1, 2]
    if not isinstance(shots, np.ndarray):
        shots = np.array(shots)
    t1 = time.time()
    metadata_path = data_dir / 'bes_metadata.hdf5'
    with h5py.File(metadata_path, 'a') as metadata_file:
        valid_shot_counter = 0
        configuration_group = metadata_file.require_group('configurations')
        config_8x8_group = configuration_group.require_group(
            '8x8_configurations')
        config_non_8x8_group = configuration_group.require_group(
            'non_8x8_configurations')

        def validate_configuration(input_bes_data):
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
                    max_index[igroup] = np.max([max_index[igroup], config_index])
                    # test if input data matches existing configuration
                    if not np.allclose(r_position,
                                       config.attrs['r_position'],
                                       atol=0.1):
                        continue
                    if not np.allclose(z_position,
                                       config.attrs['z_position'],
                                       atol=0.1):
                        continue
                    print(f'Configuration matches index {config_index}')
                    if input_bes_data.shot not in config.attrs['shots']:
                        config.attrs['shots'] = np.append(config.attrs['shots'],
                                                          input_bes_data.shot)
                    return config_index
            print('Configuration does not match existing configuration')
            # now test for 8x8 configuration
            config_is_8x8 = True
            for i in np.arange(8):
                rdiff = np.diff(r_position[i+np.arange(8)*8])
                col_test = np.allclose(rdiff, np.zeros(rdiff.shape), atol=0.1)
                zdiff = np.diff(z_position[i*8 + np.arange(8)])
                row_test = np.allclose(zdiff, np.zeros(zdiff.shape), atol=0.1)
                config_is_8x8 = config_is_8x8 and col_test and row_test
                if not config_is_8x8:
                    break
            print(f'New configuration is 8x8: {config_is_8x8}')
            if config_is_8x8:
                new_index = max_index[0]+1
                print(f'New 8x8 config index is {new_index}')
                new_config = config_8x8_group.create_group(f'{new_index:02d}')
            else:
                new_index = max_index[1]+1
                print(f'New non-8x8 config index is {new_index}')
                new_config = config_non_8x8_group.create_group(f'{new_index:d}')
            new_config.attrs['r_position'] = r_position
            new_config.attrs['z_position'] = z_position
            new_config.attrs['shots'] = np.array([input_bes_data.shot], dtype=np.int)
            if config_is_8x8:
                new_config.attrs['r_avg'] = np.mean(r_position)
                new_config.attrs['z_avg'] = np.mean(z_position)
            return new_index

        for ishot, shot in enumerate(shots):
            print(f'Shot {shot} ({ishot + 1} of {shots.size})')
            bes_data = BES_Data(shot=shot, channels=channels, verbose=verbose)
            if bes_data.time is None:
                print(f'INVALID BES data for shot {shot}')
                continue
            shot_string = f'{bes_data.shot:d}'
            shot_group = metadata_file.require_group(shot_string)
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
            config_index = validate_configuration(bes_data)
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
            # signals
            if with_signals:
                signal_file = data_dir / f'bes_signals_{shot_string}.hdf5'
                bes_data.get_signals()
                with h5py.File(signal_file, 'w') as sfile:
                    if verbose:
                        t3 = time.time()
                    sfile.create_dataset('signals',
                                         data=bes_data.signals,
                                         compression='gzip',
                                         chunks=True)
                    sfile.create_dataset('time',
                                         data=bes_data.time,
                                         compression='gzip',
                                         chunks=True)
                    if verbose:
                        t4 = time.time()
                        print(f'Write signals elapsed time = {t4 - t3:.2f} s')
                        traverse_h5py(sfile)
    t2 = time.time()
    print_metadata_summary(path=metadata_path)
    print(f'Packaging data elapsed time = {t2 - t1:.2f} s')
    print(f'{valid_shot_counter} valid shots out of {shots.size} in input shot list')


def print_metadata_summary(path=None):
    if not path:
        path = data_dir / 'bes_metadata.hdf5'
    if not isinstance(path, Path):
        path = Path(path)
    print(f'Summarizing metadata file {path.as_posix()}')
    with h5py.File(path, 'r') as metadata_file:
        traverse_h5py(metadata_file)
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        config_non_8x8_group = metadata_file['configurations']['non_8x8_configurations']
        for group in [config_8x8_group, config_non_8x8_group]:
            sum_shots = 0
            for config_group in group.values():
                nshots = config_group.attrs['shots'].size
                sum_shots += nshots
                print(f'# of shots in {config_group.name}: {nshots}')
            print(f'  Sum of shots in {group.name} group: {sum_shots}')

if __name__ == '__main__':
    shotlist = [176778, 171472, 171473, 171477, 171495,
                145747, 145745, 142300, 142294, 145384]
    package_bes(shots=shotlist, verbose=True, with_signals=False)
