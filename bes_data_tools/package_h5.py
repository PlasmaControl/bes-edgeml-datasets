from pathlib import Path
import csv
import threading
import concurrent.futures
import time
import os
import numpy as np
import h5py

try:
    from .bes_data import BES_Data
except ImportError:
    try:
        from bes_data_tools.bes_data import BES_Data
    except ImportError:
        from bes_data import BES_Data


# make standard directories
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)


def _config_data_file(data_file, must_exist=True):
    data_file = Path(data_file)
    if 'data' not in data_file.parent.as_posix():
        data_file = data_dir / data_file
    if must_exist:
        if not data_file.exists():
            raise FileNotFoundError(f'File {data_file.as_posix()} not found.')
    return data_file


def print_h5py_contents(input_hdf5file=None,
                        skip_subgroups=False):
    input_hdf5file = _config_data_file(input_hdf5file)
    print(f'Summarizing HDF5 file {input_hdf5file.as_posix()}')
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
        print(f'Group {group.name}')
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                if skip_subgroups:
                    continue
                recursively_print_content(value)
            if isinstance(value, h5py.Dataset):
                print(f'  Dataset {key}:', value.shape, value.dtype)
                print_attributes(value)
        print_attributes(group)

    # the file object functions like a group
    # it is the top-level group, known as `root` or `/`
    print(f'Contents of {input_hdf5file.as_posix()}')
    with h5py.File(input_hdf5file.as_posix(), 'r') as file:
        # loop over key/value pairs at file root;
        # values may be a group or dataset
        recursively_print_content(file)


def print_metadata_contents(input_hdf5file=None,
                            only_8x8=False):
    input_hdf5file = _config_data_file(input_hdf5file)
    print(f'Summarizing metadata file {input_hdf5file.as_posix()}')
    with h5py.File(input_hdf5file, 'r') as metadata_file:
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        config_non_8x8_group = metadata_file['configurations']['non_8x8_configurations']
        if only_8x8:
            print_h5py_contents(config_8x8_group)
        else:
            print_h5py_contents(input_hdf5file)
        for group in [config_8x8_group, config_non_8x8_group]:
            sum_shots = 0
            for config_group in group.values():
                nshots = config_group.attrs['shots'].size
                sum_shots += nshots
                print(f'# of shots in {config_group.name}: {nshots}')
            print(f'Sum of shots in {group.name} group: {sum_shots}')
            if only_8x8:
                break


def _validate_configuration(bes_data,
                            config_8x8_group,
                            config_non_8x8_group):
    config_indices = np.array([0, 1000])
    r_position = bes_data.metadata['r_position']
    z_position = bes_data.metadata['z_position']
    for igroup, config_group in enumerate([config_8x8_group,
                                           config_non_8x8_group]):
        for config_index_str, config in config_group.items():
            config_index = int(config_index_str)
            assert ('r_position' in config.attrs and
                    'z_position' in config.attrs and
                    'shots' in config.attrs)
            config_indices[igroup] = np.max(
                [config_indices[igroup], config_index]
            )
            # test if input data matches existing configuration
            if not np.allclose(r_position,
                               config.attrs['r_position'],
                               atol=0.1):
                continue
            if not np.allclose(z_position,
                               config.attrs['z_position'],
                               atol=0.1):
                continue
            print(f'{bes_data.shot}: Configuration matches index {config_index}')
            if bes_data.shot not in config.attrs['shots']:
                config.attrs['shots'] = np.append(config.attrs['shots'],
                                                  bes_data.shot)
                config.attrs['nshots'] = config.attrs['shots'].size
            return config_index
    # now test for 8x8 configuration
    config_is_8x8 = bes_data.metadata['is_8x8']
    if config_is_8x8:
        new_index = config_indices[0] + 1
        print(f'{bes_data.shot}: New 8x8 config index is {new_index}')
        new_config = config_8x8_group.create_group(f'{new_index:04d}')
        new_config.attrs['r_avg'] = bes_data.metadata['r_avg']
        new_config.attrs['z_avg'] = bes_data.metadata['z_avg']
        z_first_column = z_position[np.arange(8) * 8]
        new_config.attrs['upper_inboard_channel'] = z_first_column.argmax() * 8
        new_config.attrs['lower_inboard_channel'] = z_first_column.argmin() * 8
    else:
        new_index = config_indices[1] + 1
        print(f'{bes_data.shot}: New non-8x8 config index is {new_index}')
        new_config = config_non_8x8_group.create_group(f'{new_index:04d}')
    new_config.attrs['r_position'] = r_position
    new_config.attrs['z_position'] = z_position
    new_config.attrs['shots'] = np.array([bes_data.shot], dtype=int)
    new_config.attrs['nshots'] = new_config.attrs['shots'].size
    return new_index


def _validate_bes_data(shot=None,
                       channels=None,
                       verbose=False,
                       with_signals=False,
                       metafile=None,
                       lock=None,
                       only_8x8=True,
                       only_standard_8x8=True,
                       ):

    bes_data = BES_Data(
            shot=shot,
            channels=channels,
            verbose=verbose,
            get_signals=with_signals,
            only_8x8=only_8x8,
            only_standard_8x8=only_standard_8x8,
    )
    if bes_data.time is None:
        print(f'{bes_data.shot}: ERROR invalid BES_Data object')
        return -1
    shot_string = f'{bes_data.shot:d}'
    # metadata attributes
    if lock:
        lock.acquire()
    configuration_group = metafile.require_group('configurations')
    config_8x8_group = configuration_group.require_group('8x8_configurations')
    config_non_8x8_group = configuration_group.require_group('non_8x8_configurations')
    config_index = _validate_configuration(bes_data,
                                           config_8x8_group,
                                           config_non_8x8_group)
    if only_8x8:
        if config_index >= 1000:
            print(f'{bes_data.shot}: ERROR not 8x8 config')
            return -1
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
        signals_dir = data_dir / 'signals'
        signals_dir.mkdir(exist_ok=True)
        if bes_data.signals is None:
            print(f'{bes_data.shot}: ERROR invalid BES signals')
            return -1
        signal_file = signals_dir / f'bes_signals_{shot_string}.hdf5'
        with h5py.File(signal_file.as_posix(), 'w') as sfile:
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
    return shot


def package_bes(
        shotlist=[196554,196555,196559,196560],
        input_csvfile=None,
        max_shots=None,
        output_hdf5file='metadata.hdf5',
        channels=None,
        verbose=False,
        with_signals=False,
        max_workers=2,
        use_concurrent=False,
        only_8x8=True,
        only_standard_8x8=True,
):
    if input_csvfile:
        # override `shotlist`
        # use CSV file with 'shot' column to create `shotlist`
        input_csvfile = _config_data_file(input_csvfile)
        print(f'Using shotlist {input_csvfile.as_posix()}')
        shotlist = []
        with input_csvfile.open() as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            assert 'shot' in reader.fieldnames
            for irow, row in enumerate(reader):
                if max_shots and irow+1 > max_shots:
                    break
                shotlist.append(int(row['shot']))
    shotlist = np.array(shotlist, dtype=int)
    if channels is None:
        channels = np.arange(1,65)
    channels = np.array(channels, dtype=int)
    output_hdf5file = _config_data_file(output_hdf5file, must_exist=False)
    t1 = time.time()
    with h5py.File(output_hdf5file.as_posix(), 'w') as h5file:
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
                                             lock=lock,
                                             only_8x8=only_8x8,
                                             only_standard_8x8=only_standard_8x8,
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
                result = _validate_bes_data(
                        shot=shot,
                        channels=channels,
                        verbose=verbose,
                        with_signals=with_signals,
                        metafile=h5file,
                        only_8x8=only_8x8,
                        only_standard_8x8=only_standard_8x8,
                )
                if result and result>0:
                    valid_shot_counter += 1
                else:
                    print(f'{shot}: INVALID return value')
    t2 = time.time()
    if verbose:
        print_metadata_contents(input_hdf5file=output_hdf5file)
    dt = t2 - t1
    print(f'Packaging data elapsed time: {int(dt)//3600} hr {dt%3600/60:.1f} min')
    print(f'{valid_shot_counter} valid shots out of {shotlist.size} in input shot list')

def make_8x8_sublist(input_hdf5file='sample_metadata.hdf5',
                     upper_inboard_channel=None,
                     verbose=False,
                     r_range=(223, 227),
                     z_range=(-1.5, 1)):
    input_hdf5file = _config_data_file(input_hdf5file)
    shotlist = np.array((), dtype=int)
    with h5py.File(input_hdf5file, 'r') as metadata_file:
        config_8x8_group = metadata_file['configurations']['8x8_configurations']
        for name, config in config_8x8_group.items():
            upper = config.attrs['upper_inboard_channel']
            if upper_inboard_channel is not None and upper != upper_inboard_channel:
                continue
            shots = config.attrs['shots']
            r_avg = config.attrs['r_avg']
            z_avg = config.attrs['z_avg']
            z_position = config.attrs['z_position']
            delta_z = z_position.max() - z_position.min()
            valid = r_range[0] <= r_avg <= r_range[1] and \
                    z_range[0] <= z_avg <= z_range[1] and \
                    delta_z <= 12
            if not valid:
                continue
            shotlist = np.append(shotlist, shots)
            # nshots.append(shots.size)
            # r.append(r_avg)
            # z.append(z_avg)
            if verbose:
                print(f'8x8 config #{name} nshots {shots.size} ravg {r_avg:.2f} upper {upper}')
    print(f'Shots within r/z min/max limits: {shotlist.size}')
    return shotlist


def package_8x8_signals(input_hdf5file='sample_metadata.hdf5',
                        max_shots=None,
                        output_hdf5file='sample_metadata_8x8.hdf5',
                        channels=None):
    input_hdf5file = _config_data_file(input_hdf5file)
    print(f'Using metadata in {input_hdf5file.as_posix()}')
    shotlist = make_8x8_sublist(
            input_hdf5file=input_hdf5file,
            upper_inboard_channel=56)
    if max_shots:
        shotlist = shotlist[0:max_shots]
    package_bes(shotlist=shotlist,
                output_hdf5file=output_hdf5file,
                verbose=True,
                with_signals=True,
                channels=channels,
                )

def downsample_elm_dataset(file=None):
    assert(file is not None)
    file = Path(file)
    assert(file.exists())
    newfile = file.parent / f'{file.stem}-small.hdf5'
    with h5py.File(file, 'r') as source_h5, \
        h5py.File(newfile, 'w') as small_h5:
        group_names = list(source_h5.keys())
        for group_name in group_names[::10]:
            group = source_h5[group_name]
            assert(isinstance(group, h5py.Group))
            print(f'Copying group: {group.name}')
            new_group = small_h5.create_group(group.name)
            for dataset in group.values():
                assert(isinstance(dataset, h5py.Dataset))
                print(f'  Copying dataset: {dataset.name}')
                new_group.create_dataset(dataset.name, data=dataset[:])
            for attrname, attrvalue in group.attrs.items():
                print(f'  Copying attribute: {attrname}')
                new_group.attrs[attrname] = attrvalue




def read_metadata(input_hdf5file='sample_metadata.hdf5',
                  verbose=False):
    input_hdf5file = _config_data_file(input_hdf5file)
    with h5py.File(input_hdf5file.as_posix(), 'r') as hfile:
        configuration_group = hfile['configurations']
        metadata = []
        for name, group in hfile.items():
            if group == configuration_group:
                continue
            attrs = group.attrs
            string = f"{attrs['shot']:6d}    " + \
                     f"{attrs['date']:23}    " + \
                     f"tstart {attrs['start_time']:6.1f}  " + \
                     f"tstop {attrs['stop_time']:6.1f}  "
            print(string)


def dataset_stats(
        hdf5_file='metadata.hdf5',
):
    hdf5_file = _config_data_file(hdf5_file, must_exist=True)
    print(f"HDF5 file: {hdf5_file}")
    ip_cw_bt_cw = ip_ccw_bt_cw = ip_cw_bt_ccw = ip_ccw_bt_ccw = 0
    with h5py.File(hdf5_file, mode='r') as h5:
        print(f"Top-level groups: {len(h5)}")
        for group_key, group in h5.items():
            if group_key.startswith('config'):
                continue
            if group.attrs['ip'] > 0:
                if group.attrs['bt'] > 0:
                    ip_cw_bt_cw += 1
                else:
                    ip_cw_bt_ccw += 1
            else:
                if group.attrs['bt'] > 0:
                    ip_ccw_bt_cw += 1
                else:
                    ip_ccw_bt_ccw += 1
    print(f"Ip CW, Bt CW: {ip_cw_bt_cw}")
    print(f"Ip CW, Bt CCW: {ip_cw_bt_ccw}")
    print(f"Ip CCW, Bt CW: {ip_ccw_bt_cw}")
    print(f"Ip CCW, Bt CCW: {ip_ccw_bt_ccw}")
    return


if __name__=='__main__':
    package_bes(
            # input_csvfile='big_shotlist.csv',
            max_shots=5,
            verbose=True,
            # use_concurrent=True,
    )
    # dataset_stats(
    #         hdf5_file='/home/smithdr/edgeml/elm/data/big_metadata.hdf5',
    # )
