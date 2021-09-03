"""
Functions to package BES metadata and signals in HDF5 files.

`package_bes` is the primary function that packages metadata (signals are
optional).

`package_8x8_signals` filters metadata and saves HDF5 files for signals and
filtered metadata.
"""

from pathlib import Path
import csv
import threading
import concurrent
import time
import os
import numpy as np
import h5py

try:
    from .bes_data import BES_Data
except ImportError:
    from bes_data_tools.bes_data import BES_Data


# make standard directories
Path('data').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)


def print_h5py_contents(input_filename, skip_subgroups=False):
    # private function to print attributes, if any
    # groups or datasets may have attributes
    def print_attributes(obj):
        for key, value in obj.attrs.items():
            if isinstance(value, np.ndarray):
                print(f'  Attribute {key}:', value.shape, value.dtype)
            else:
                print(f'  Attribute {key}:', value)

    # private function to recursively print groups/subgroups and datasets
    def recursively_print_content(group):
        # loop over items in a group
        # items may be subgroup or dataset
        # items are key/value pairs
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                if skip_subgroups:
                    continue
                recursively_print_content(value)
            if isinstance(value, h5py.Dataset):
                print(f'  Dataset {key}:', value.shape, value.dtype)
                print_attributes(value)
        print(f'Group {group.name}')
        print_attributes(group)

    # the file object functions like a group
    # it is the top-level group, known as `root` or `/`
    print(f'Contents of {input_filename}')
    with h5py.File(input_filename, 'r') as file:
        # loop over key/value pairs at file root;
        # values may be a group or dataset
        recursively_print_content(file)


def print_metadata_contents(path=None, only_8x8=False):
    if not path:
        path = '../elms/data/bes_metadata.hdf5'
    if not isinstance(path, Path):
        path = Path(path)
    print(f'Summarizing metadata file {path.as_posix()}')
    with h5py.File(path, 'r') as metadata_file:
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        config_non_8x8_group = metadata_file['configurations']['non_8x8_configurations']
        if only_8x8:
            print_h5py_contents(config_8x8_group)
        else:
            print_h5py_contents(path)
        for group in [config_8x8_group, config_non_8x8_group]:
            sum_shots = 0
            for config_group in group.values():
                nshots = config_group.attrs['shots'].size
                sum_shots += nshots
                print(f'# of shots in {config_group.name}: {nshots}')
            print(f'Sum of shots in {group.name} group: {sum_shots}')
            if only_8x8:
                break


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


def _validate_bes_data(shot=None,
                       channels=None,
                       verbose=False,
                       with_signals=False,
                       metafile=None,
                       lock=None):

    bes_data = BES_Data(shot=shot,
                        channels=channels,
                        verbose=verbose,
                        get_signals=with_signals)
    if bes_data.time is None:
        print(f'{bes_data.shot}: ERROR invalid BES_Data object')
        return -bes_data.shot
    shot_string = f'{bes_data.shot:d}'
    # signals
    if with_signals:
        Path('data/signals').mkdir(parents=True, exist_ok=True)
        if bes_data.signals is None:
            print(f'{bes_data.shot}: ERROR invalid BES signals')
            return -bes_data.shot
        signal_file = f'data/signals/bes_signals_{shot_string}.hdf5'
        with h5py.File(signal_file, 'w') as sfile:
            sfile.create_dataset('signals',
                                 data=bes_data.signals,
                                 compression='gzip',
                                 chunks=True)
            sfile.create_dataset('time',
                                 data=bes_data.time,
                                 compression='gzip',
                                 chunks=True)
        if verbose:
            print_h5py_contents(signal_file)
        signal_mb = bes_data.signals.nbytes // 1024 // 1024
        print(f'{bes_data.shot}: BES_Data size = {signal_mb} MB')
    # metadata attributes
    if lock:
        lock.acquire()
    configuration_group = metafile.require_group('configurations')
    config_8x8_group = configuration_group.require_group('8x8_configurations')
    config_non_8x8_group = configuration_group.require_group('non_8x8_configurations')
    shot_group = metafile.require_group(shot_string)
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
    del bes_data
    return shot


def package_bes(shotlist=(176778, 171472),
                input_csvfile=None,
                max_shots=None,
                output_h5file='data/sample_metadata.hdf5',
                channels=None,
                verbose=False,
                with_signals=False,
                max_workers=2,
                use_concurrent=False):
    if input_csvfile:
        # override `shotlist`
        # use CSV file with 'shot' column to create `shotlist`
        input_csvfile = Path(input_csvfile)
        print(f'Using shotlist {input_csvfile.as_posix()}')
        assert(input_csvfile.exists())
        shotlist = []
        with input_csvfile.open() as csvfile:
            reader = csv.DictReader(csvfile,
                                    fieldnames=None,
                                    skipinitialspace=True)
            for irow, row in enumerate(reader):
                if max_shots and irow+1 > max_shots:
                    break
                shotlist.append(int(row['shot']))
    shotlist = np.array(shotlist)
    if channels is None:
        channels = np.arange(1,65)
    channels = np.array(channels)
    output_h5file = Path(output_h5file)
    t1 = time.time()
    with h5py.File(output_h5file.as_posix(), 'w') as h5file:
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
                    future = executor.submit(_validate_bes_data,
                                             shot=shot,
                                             channels=channels,
                                             verbose=verbose,
                                             with_signals=with_signals,
                                             metafile=h5file,
                                             lock=lock)
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
                shot = _validate_bes_data(shot=shot,
                                          channels=channels,
                                          verbose=verbose,
                                          with_signals=with_signals,
                                          metafile=h5file)
                if shot and shot>0:
                    valid_shot_counter += 1
                    print( f'{shot} good')
                else:
                    print(f'{-shot} INVALID return value')
    t2 = time.time()
    if verbose:
        print_metadata_contents(path=output_h5file)
    dt = t2 - t1
    print(f'Packaging data elapsed time: {int(dt)//3600} hr {dt%3600/60:.1f} min')
    print(f'{valid_shot_counter} valid shots out of {shotlist.size} in input shot list')

def make_8x8_sublist(input_h5file='data/sample_metadata.hdf5',
                     upper_inboard_channel=None,
                     verbose=False,
                     r_range=(223, 227),
                     z_range=(-1.5, 1)):
    input_h5file = Path(input_h5file)
    # r = []
    # z = []
    # nshots = []
    shotlist = np.array((), dtype=np.int)
    with h5py.File(input_h5file, 'r') as metadata_file:
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        for name, config in config_8x8_group.items():
            upper = config.attrs['upper_inboard_channel']
            if upper_inboard_channel is not None and upper != upper_inboard_channel:
                continue
            shots = config.attrs['shots']
            r_avg = config.attrs['r_avg']
            z_avg = config.attrs['z_avg']
            z_position = config.attrs['z_position']
            # nshots.append(shots.size)
            # r.append(r_avg)
            # z.append(z_avg)
            delta_z = z_position.max() - z_position.min()
            valid_condition = \
                r_range[0] <= r_avg <= r_range[1] and \
                z_range[0] <= z_avg <= z_range[1] and \
                delta_z <= 12
            if valid_condition:
                shotlist = np.append(shotlist, shots)
            if verbose:
                print(f'8x8 config #{name} nshots {shots.size} ravg {r_avg:.2f} upper {upper}')
    print(f'Shots within r/z min/max limits: {shotlist.size}')
    # if not noplot:
    #     plt.plot(r, z, 'x')
    #     for i, nshot in enumerate(nshots):
    #         plt.annotate(repr(nshot),
    #                      (r[i], z[i]),
    #                      textcoords='offset points',
    #                      xytext=(0,10),
    #                      ha='center')
    #     plt.xlim(220, 230)
    #     plt.ylim(-1.5, 1.5)
    #     for r in rminmax:
    #         plt.vlines(r, zminmax[0], zminmax[1], color='k')
    #     for z in zminmax:
    #         plt.hlines(z, rminmax[0], rminmax[1], color='k')
    #     plt.xlabel('R (cm)')
    #     plt.ylabel('Z (cm)')
    #     plt.title('R/Z centers of BES 8x8 grids, and shot counts')
    return shotlist


def package_8x8_signals(input_h5file='data/sample_metadata.hdf5',
                        max_shots=None,
                        output_h5file='data/sample_metadata_8x8.hdf5',
                        channels=None):
    input_h5file = Path(input_h5file)
    print(f'Using metadata in {input_h5file.as_posix()}')
    assert(input_h5file.exists())
    shot_list = make_8x8_sublist(
            input_h5file=input_h5file,
            upper_inboard_channel=56,
            noplot=True)
    if max_shots:
        shot_list = shot_list[0:max_shots]
    package_bes(shotlist=shot_list,
        output_h5file=output_h5file,
        verbose=True,
        with_signals=True,
        channels=channels,
        )


if __name__=='__main__':
    # get metadata from shotlist or csv
    package_bes(input_csvfile='data/sample_shotlist.csv',
                max_shots=2,
                output_h5file='data/sample_metadata.hdf5',
                verbose=True)
    # filter metadata and save signals
    package_8x8_signals(input_h5file='data/sample_metadata.hdf5',
                        output_h5file='data/sample_metadata_8x8.hdf5',
                        channels=[1,2])