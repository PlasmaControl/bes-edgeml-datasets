from pathlib import Path
import csv
import numpy as np
from edgeml import bes2hdf5
import h5py
import time as ttime

directory = Path(__file__).parent

def read_shotlist(filename=None):
    if not filename:
        filename = directory / 'shotlist.csv'
    filename = Path(filename)
    assert(filename.exists())
    shotlist = []
    with filename.open() as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=None,
                                skipinitialspace=True)
        for irow, row in enumerate(reader):
            shotlist.append(int(row['shot']))
    return np.array(shotlist)


def package_shotlist_metadata(max_shots=None):
    shotlist = read_shotlist()
    if max_shots:
        shotlist = shotlist[0:max_shots]
    bes2hdf5.package_bes(shots=shotlist,
                         verbose=True,
                         with_signals=False)


def package_8x8_sublist(max_shots=None):
    metadata_file = directory / 'data/metadata_full_shotlist/bes_metadata.hdf5'
    shotlist = bes2hdf5.make_8x8_sublist(
            path=metadata_file,
            upper_inboard_channel=56,
            noplot=True)
    if max_shots:
        shotlist = shotlist[0:max_shots]
    bes2hdf5.package_bes(shots=shotlist,
                         verbose=True,
                         with_signals=True)


def package_elm_events(max_elms=None, verbose=False):
    elm_file = directory / 'elm-list.csv'
    elms = []
    t1 = ttime.time()
    with elm_file.open() as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=None,
                                skipinitialspace=True)
        for irow, row in enumerate(reader):
            elms.append({'shot': int(row['shot']),
                         'start_time': float(row['start_time']),
                         'stop_time': float(row['stop_time'])})
    elm_file = directory / 'elm-events.hdf5'
    signals_directory = directory / 'data/signals_8x8_only'
    current_shot = 0
    signal_group = None
    failed_shots = []
    with h5py.File(elm_file, 'w') as elm_file_group:
        for ielm, elm in enumerate(elms):
            if max_elms and ielm > max_elms:
                break
            if elm['shot'] != current_shot:
                signal_file = signals_directory / f"bes_signals_{elm['shot']:d}.hdf5"
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
                signals = signal_group['signals']
                time = signal_group['time']
                print('  Finished loading')
            dt = elm['stop_time'] - elm['start_time']
            time_mask = np.logical_and(time[:] >= (elm['start_time'] - dt),
                                       time[:] <= (elm['stop_time']  + dt))
            print(f"  ielm {ielm} in shot {current_shot} at time {elm['start_time']:.2f} ms with {np.count_nonzero(time_mask)} time points")
            elm_event_group = elm_file_group.create_group(f'{ielm:05d}')
            elm_event_group.attrs['shot'] = elm['shot']
            elm_event_group.create_dataset('signals', data=signals[:,time_mask])
            elm_event_group.create_dataset('time', data=time[time_mask])
    if signal_group:
        signal_group.close()
    if verbose:
        bes2hdf5.traverse_h5py(elm_file)
    print('Failed shots:', failed_shots)
    t2 = ttime.time()
    dt = t2 - t1
    print(f'Elapsed time: {int(dt) // 3600} hr {dt % 3600 / 60:.1f} min')


if __name__=='__main__':
    # shotlist = read_shotlist()
    # package_shotlist_metadata(max_shots=4)
    # bes2hdf5.print_metadata_summary('data/elm_metadata/bes_metadata.hdf5', only_8x8=True)
    # shotlist = bes2hdf5.make_8x8_sublist(path='data/elm_metadata/bes_metadata.hdf5',
    #                                      upper_inboard_channel=56)
    # package_8x8_sublist()
    package_elm_events(verbose=True)
