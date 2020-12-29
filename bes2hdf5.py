from pathlib import Path
import time
import h5py
import numpy as np
import MDSplus

connection = MDSplus.Connection('atlas.gat.com')

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)


def traverse_h5py(group):
    def print_attrs(obj):
        for attr_name, attr_value in obj.attrs.items():
            print(f'TRH5:   Attribute: {attr_name} {attr_value}')
    print(f'TRH5: Group {group.name} in file {group.file}')
    print_attrs(group)
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            traverse_h5py(value)
        if isinstance(value, h5py.Dataset):
            print(f'TRH5:   Dataset {value.name}', value.shape, value.dtype)
            print_attrs(value)


big_shotlist = [176778, 171472, 171473, 171477, 171495,
                145747, 145745, 142300, 142294, 145384]


class BES_Data(object):
    _points = ['ip',
               'bt',
               # 'density',
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
        self.verbose = verbose
        if self.verbose:
            print(f'Getting time and metadata for shot {self.shot}')
            t1 = time.time()
        # get time array
        ptdata = f'ptdata("besfu01", {self.shot})'
        self.time = np.array(connection.get(f'dim_of({ptdata})'))
        n_time = connection.get(f'size({ptdata})')
        self.n_time = n_time.data()
        assert(self.n_time == self.time.size)
        # get metadata
        connection.openTree('bes', self.shot)
        r_position = np.array(connection.get(r'\bes_r')).round(decimals=2)
        z_position = np.array(connection.get(r'\bes_z')).round(decimals=2)
        start_time = connection.get(r'\bes_ts')
        connection.closeTree('bes', self.shot)
        assert(start_time == self.time[0])
        self.metadata = {'shot': self.shot,
                         'delta_time': np.diff(self.time[0:100]).mean().round(decimals=4),
                         'start_time': self.time[0],
                         'stop_time': self.time[-1],
                         'n_time': self.n_time,
                         'time_units': 'ms',
                         'r_position': r_position,
                         'z_position': z_position,
                         'rz_units': 'cm'}
        # get ip, beams, etc.
        for point_name in self._points:
            data = np.array(0)
            data_time = np.array(0)
            try:
                if 'inj' in point_name:
                    connection.openTree('nb', self.shot)
                    data = np.array(connection.get(f'\\{point_name}'))
                    data_time = np.array(connection.get(f'dim_of(\\{point_name})'))
                    connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(connection.get(ptdata))
                    data_time = np.array(connection.get('dim_of(_n)'))
                time_mask = np.logical_and(data_time>=self.time[0], data_time<=self.time[-1])
                data = data[time_mask]
                data_time = data_time[time_mask]
            except:
                print(f'INVALID data node for shot {self.shot}: {point_name}')
                data = h5py.Empty(dtype='f')
                data_time = h5py.Empty(dtype='f')
            assert (data.shape == data_time.shape)
            setattr(self, point_name, data)
            if point_name in ['ip','bt','pinj']:
                setattr(self, f'{point_name}_time', data_time)
        if self.verbose:
            t2 = time.time()
            print(f'  Shot {self.shot} with {self.n_time} time points')
            print(f'  Time, metadata elapsed time = {t2-t1:.2f} s')

    def get_signals(self):
        tdi_vars = []
        tdi_assignments = []
        for channel in self.channels:
            tdi_vars.append(f'_n{channel:02d}')
            tdi_assignments.append(f'{tdi_vars[-1]} = ptdata("besfu{channel:02d}", {self.shot})')
        if self.verbose:
            print(f'  Fetching signals ({self.channels.size} channels) for shot {self.shot}')
            t1 = time.time()
        connection.get(', '.join(tdi_assignments))
        self.signals = np.empty([self.channels.size, self.n_time])
        for i, tdi_var in enumerate(tdi_vars):
            self.signals[i, :] = connection.get(tdi_var)
        if self.verbose:
            t2 = time.time()
            print(f'  Get signals elapsed time = {t2-t1:.2f} s')


def package_bes_data(shots=None, channels=None, verbose=False, with_signals=False):
    if not shots and not channels:
        shots = [176778, 171472]
        channels = [1, 2]
    if not isinstance(shots, np.ndarray):
        shots = np.array(shots)
    meta_file = data_dir / 'bes_metadata.hdf5'
    with h5py.File(meta_file, 'a') as mfile:
        t1 = time.time()
        for shot in shots:
            bes_data = BES_Data(shot=shot, channels=channels, verbose=verbose)
            shot_string = f'{bes_data.shot:d}'
            mgroup = mfile.require_group(shot_string)
            # metadata attributes
            for attr_name, attr_value in bes_data.metadata.items():
                if attr_name in mgroup.attrs:
                    if 'position' in attr_name:
                        assert(np.allclose(attr_value,
                                           mgroup.attrs[attr_name],
                                           atol=0.1))
                    else:
                        assert(attr_value == mgroup.attrs[attr_name])
                else:
                    mgroup.attrs[attr_name] = attr_value
            # metadata datasets
            for point_name in bes_data._points:
                for name in [f'{point_name}', f'{point_name}_time']:
                    data = getattr(bes_data, name, None)
                    if data is None:
                        continue
                    mgroup.require_dataset(name,
                                           data=data,
                                           shape=data.shape,
                                           dtype=data.dtype,
                                           compression='gzip')
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
        if verbose:
            print('Metadata file')
            traverse_h5py(mfile)
        t2 = time.time()
        print(f'Packaging data elapsed time = {t2 - t1:.2f} s')

def small_job():
    package_bes_data(verbose=True, with_signals=True)

def big_job():
    package_bes_data(shots=big_shotlist, verbose=True, with_signals=True)

if __name__ == '__main__':
    small_job()
