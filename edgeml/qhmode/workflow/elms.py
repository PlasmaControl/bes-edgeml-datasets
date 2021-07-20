from pathlib import Path
import csv
import numpy as np
from edgeml import bes2hdf5
import h5py
import time as timelib


# establish directories and CSV files in `elms/data`
data_directory = Path(__file__).parent / 'data'
data_directory.mkdir(exist_ok=True)
metadata_directory = data_directory / 'metadata'
metadata_directory.mkdir(exist_ok=True)
signals_directory = data_directory / 'signals-8x8-only'
signals_directory.mkdir(exist_ok=True)
unlabeled_elms_directory = data_directory / 'unlabeled-elm-events'
unlabeled_elms_directory.mkdir(exist_ok=True)
shot_list_file = data_directory / 'shotlist.csv'
elm_list_file = data_directory / 'elm-list.csv'


def package_metadata(max_shots=None):
    print(f'Using shotlist {shot_list_file.as_posix()}')
    assert(shot_list_file.exists())
    shot_list = []
    with shot_list_file.open() as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=None,
                                skipinitialspace=True)
        for irow, row in enumerate(reader):
            shot_list.append(int(row['shot']))
            if max_shots and irow > max_shots:
                break
    # filename = Path('bes_metadata.hdf5')
    bes2hdf5.package_bes(shots=shot_list,
                         verbose=True,
                         with_signals=False,
                         filename='bes_metadata.hdf5',
                         )


def package_signals_8x8_only(max_shots=None):
    metadata_file = metadata_directory / 'bes_metadata.hdf5'
    print(f'Using metadata in {metadata_file.as_posix()}')
    assert(metadata_file.exists())
    shot_list = bes2hdf5.make_8x8_sublist(
            path=metadata_file,
            upper_inboard_channel=56,
            noplot=True)
    if max_shots:
        shot_list = shot_list[0:max_shots]
    bes2hdf5.package_bes(shots=shot_list,
                         verbose=True,
                         with_signals=True)


def package_unlabeled_elm_events(max_elms=None):
    print(f'Using ELM list file {elm_list_file.as_posix()}')
    assert(elm_list_file.exists())
    elms = []
    t1 = timelib.time()
    with elm_list_file.open() as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=None,
                                skipinitialspace=True)
        for irow, row in enumerate(reader):
            elms.append({'shot': int(row['shot']),
                         'start_time': float(row['start_time']),
                         'stop_time': float(row['stop_time'])})
    unlabeled_elm_events_file = Path('elm-events.hdf5')  # local output file
    metadata_file = signals_directory / 'bes_metadata.hdf5'
    current_shot = 0
    signal_group = None
    failed_shots = []
    with h5py.File(unlabeled_elm_events_file, 'w') as elm_file_group, \
            h5py.File(metadata_file, 'r') as metafile:
        for ielm, elm in enumerate(elms):
            if max_elms and ielm > max_elms:
                break
            if elm['shot'] != current_shot:
                signal_file = signals_directory / \
                              f"bes_signals_{elm['shot']:d}.hdf5"
                if not signal_file.exists():
                    if elm['shot'] not in failed_shots:
                        print(f'File {signal_file.as_posix()} does not exist !!!')
                        failed_shots.append(elm['shot'])
                    continue
                current_shot = elm['shot']
                if signal_group:
                    signal_group.close()
                print(f"Loading shot {elm['shot']}")
                signal_group = h5py.File(signal_file, 'r')
                pinj_time = metafile[f'{current_shot:d}']['pinj_time'][:]
                pinj_15l = metafile[f'{current_shot:d}']['pinj_15l'][:]
                pinj_15r = metafile[f'{current_shot:d}']['pinj_15r'][:]
                signals = signal_group['signals'][:, :]
                time = signal_group['time'][:]
                print('  Finished loading')
            dt = elm['stop_time'] - elm['start_time']
            # skip ELM event if PING 15L/R too low/high
            pinj_mask = np.logical_and(pinj_time >= (elm['start_time'] - dt),
                                       pinj_time <= (elm['stop_time'] + dt))
            if (np.max(pinj_15l[pinj_mask]) < 0.5e6) or \
                    (np.max(pinj_15r[pinj_mask]) > 0.5e6):
                print(f'Skipping ELM {ielm} with max pinj_15l {np.max(pinj_15l[pinj_mask])} ' + \
                      f'and max pinj_15r {np.max(pinj_15r[pinj_mask])}')
                continue
            time_mask = np.logical_and(time >= (elm['start_time'] - dt),
                                       time <= (elm['stop_time'] + dt))
            print(f"  ielm {ielm} in shot {current_shot} at time {elm['start_time']:.2f} ms " + \
                  f"with {np.count_nonzero(time_mask)} time points")
            elm_event_group = elm_file_group.create_group(f'{ielm:05d}')
            elm_event_group.attrs['shot'] = elm['shot']
            elm_event_group.create_dataset('signals', data=signals[:, time_mask])
            elm_event_group.create_dataset('time', data=time[time_mask])
    if signal_group:
        signal_group.close()
    bes2hdf5.traverse_h5py(unlabeled_elm_events_file)
    print('Failed shots:', failed_shots)
    t2 = timelib.time()
    dt = t2 - t1
    print(f'Elapsed time: {int(dt) // 3600} hr {dt % 3600 / 60:.1f} min')


# if __name__ == '__main__':
    # shotlist = read_shotlist()
    # package_shotlist_metadata(max_shots=4)
    # bes2hdf5.print_metadata_summary('data/elm_metadata/bes_metadata.hdf5', only_8x8=True)
    # shotlist = bes2hdf5.make_8x8_sublist(path='data/elm_metadata/bes_metadata.hdf5',
    #                                      upper_inboard_channel=56)
    # package_8x8_sublist()
    # package_unlabeled_elm_events()
