from pathlib import Path
import h5py
import numpy as np
import MDSplus

connection = MDSplus.Connection('atlas.gat.com')

data_dir = Path.cwd() / 'data'
figures_dir = Path.cwd() / 'figures'
for d in [data_dir, figures_dir]:
    if not d.exists():
        print(f'Creating directory {d.as_posix()}')
        d.mkdir()


def get_bes(shot=176778, channels=None):
    tdi_vars = []
    tdi_assignments = []
    if not channels:
        channels = np.arange(1, 65)
    if not isinstance(channels, np.ndarray):
        channels = np.array(channels)
    for channel in channels:
        var = f'_n{channel:02d}'
        tdi_vars.append(var)
        tdi_assignments.append(var + f' = ptdata("besfu{channel:02d}", {shot})')
    print(f'Fetching data ({channels.size} channels) for shot {shot}')
    connection.get(', '.join(tdi_assignments))
    size = connection.get(f'size({tdi_vars[0]})')
    data = np.empty([channels.size, size.data()])
    for i, tdi_var in enumerate(tdi_vars):
        data[i, :] = connection.get(tdi_var)
    time = np.array(connection.get(f'dim_of({tdi_vars[0]})'))
    return data, time


def get_bes_metadata(shot=176778):
    connection.openTree('bes', shot)
    rpos = np.array(connection.get(r'\bes_r'))
    zpos = np.array(connection.get(r'\bes_z'))
    start_time = connection.get(r'\bes_ts')
    connection.closeTree('bes', shot)
    return rpos, zpos, start_time


def traverse_h5py(group):
    def print_attrs(obj):
        for attrname, attrvalue in obj.attrs.items():
            print(f'  Attribute: {attrname} {attrvalue}')

    print(f'Group {group.name} in file {group.file}')
    print_attrs(group)
    for name, value in group.items():
        if isinstance(value, h5py.Group):
            traverse_h5py(value)
        if isinstance(value, h5py.Dataset):
            print(f'  Dataset {value.name}', value.shape, value.dtype)
            print_attrs(value)


big_shotlist = [176778, 171472, 171473, 171477, 171495,
                145747, 145745, 142300, 142294, 145384]


def package_bes_data(shots=None, channels=None):
    if not shots and not channels:
        shots = [176778, 171472]
        channels = [1, 2, 3]
    if not isinstance(shots, np.ndarray):
        if not isinstance(shots, list):
            shots = [shots]
        shots = np.array(shots)
    meta_file = data_dir / 'bes_metadata.hdf5'
    with h5py.File(meta_file, 'a') as mfile:
        for shot in shots:
            data, time = get_bes(shot=shot, channels=channels)
            rpos, zpos, start_time = get_bes_metadata(shot=shot)
            assert (time[0] == start_time)
            assert (time.size == data.shape[1])
            signal_file = data_dir / f'besdata_{shot:d}.hdf5'
            with h5py.File(signal_file, 'w') as sfile:
                print(f'Saving BES data for {shot} in {signal_file.as_posix()}')
                sfile.attrs['shot'] = shot
                sfile.attrs['delta_time'] = np.diff(time[0:100]).mean()
                sfile.attrs['start_time'] = time[0]
                sfile.attrs['stop_time'] = time[-1]
                sfile.attrs['n_channels'] = data.shape[0]
                sfile.attrs['n_time'] = data.shape[1]
                sfile.attrs['time_units'] = 'ms'
                sfile.attrs['r_position'] = rpos
                sfile.attrs['z_position'] = zpos
                sfile.attrs['rz_units'] = 'cm'
                sfile.create_dataset('signals', data=data, compression='gzip')
                sfile.create_dataset('time', data=time, compression='gzip')
                traverse_h5py(sfile)


if __name__ == '__main__':
    package_bes_data()
    # rpos, zpos, start_time = get_bes_metadata()
