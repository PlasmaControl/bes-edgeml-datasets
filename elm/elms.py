from pathlib import Path
import csv
import numpy as np
import h5py
import time as timelib

from bes_data_tools.package_h5 import print_h5py_contents


def package_unlabeled_elm_events(elm_csvfile='labeled_elms.csv',
                                 max_elms=None):
    elm_csvfile = Path(elm_csvfile)
    print(f'Using ELM list file {elm_csvfile.as_posix()}')
    assert(elm_csvfile.exists())
    elms = []
    t1 = timelib.time()
    with elm_csvfile.open() as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=None,
                                skipinitialspace=True)
        for irow, row in enumerate(reader):
            elms.append({'shot': int(row['shot']),
                         'start_time': float(row['start_time']),
                         'stop_time': float(row['stop_time'])})
    unlabeled_elm_events_file = Path('elm-events.hdf5')  # local output file
    metadata_file = 'bes_metadata.hdf5'
    current_shot = 0
    signal_group = None
    failed_shots = []
    with h5py.File(unlabeled_elm_events_file, 'w') as elm_file_group, \
            h5py.File(metadata_file, 'r') as metafile:
        for ielm, elm in enumerate(elms):
            if max_elms and ielm > max_elms:
                break
            if elm['shot'] != current_shot:
                signal_file = f"bes_signals_{elm['shot']:d}.hdf5"
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
    print_h5py_contents(unlabeled_elm_events_file)
    print('Failed shots:', failed_shots)
    t2 = timelib.time()
    dt = t2 - t1
    print(f'Elapsed time: {int(dt) // 3600} hr {dt % 3600 / 60:.1f} min')
