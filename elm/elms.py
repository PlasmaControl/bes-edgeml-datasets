from pathlib import Path
import csv
import numpy as np
import h5py
import time as timelib

from bes_data_tools.package_h5 import print_h5py_contents


def package_unlabeled_elm_events(elm_csvfile='data/step_4_elm_list.csv',
                                 max_elms=None):
    elm_csvfile = Path(elm_csvfile).resolve()
    print(f'Using ELM list file {elm_csvfile.as_posix()}')
    assert elm_csvfile.exists()
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
    unlabeled_elm_events_file = Path('data/step_5_elm-events-long-windows.hdf5')  # local output file
    unlabeled_elm_events_file.unlink(missing_ok=True)
    metadata_file = Path('data/step_2_metadata.hdf5').resolve()
    assert metadata_file.exists()
    current_shot = 0
    signal_group = None
    failed_shots = []
    with h5py.File(unlabeled_elm_events_file, 'w') as elm_file_group, \
            h5py.File(metadata_file, 'r') as metafile:
        for ielm, elm in enumerate(elms):
            if max_elms and ielm > max_elms:
                break
            if elm['shot'] != current_shot:
                signal_file = Path(f"data/signals-8x8-only/bes_signals_{elm['shot']:d}.hdf5").resolve()
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
            pinj_mask = np.logical_and(pinj_time >= (elm['start_time'] - dt-10),
                                       pinj_time <= (elm['stop_time'] + dt+5))
            if (np.max(pinj_15l[pinj_mask]) < 0.5e6) or \
                    (np.max(pinj_15r[pinj_mask]) > 0.5e6):
                print(f'Skipping ELM {ielm} with max pinj_15l {np.max(pinj_15l[pinj_mask])} ' + \
                      f'and max pinj_15r {np.max(pinj_15r[pinj_mask])}')
                continue
            time_mask = np.logical_and(time >= (elm['start_time'] - dt - 10),
                                       time <= (elm['stop_time'] + dt + 5))
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


def add_elm_events(dry_run=True):
    current_file = Path('step_6_labeling_tool_v2/step_6_labeled_elm_events.hdf5').resolve()
    assert current_file.exists()
    print(f"Current file: {current_file}")

    old_file = Path('postprocess/data/labeled-elm-events-2022-01-27.hdf5').resolve()
    assert old_file.exists()
    print(f"Old file: {old_file}")

    with h5py.File(current_file, 'a') as cf, \
            h5py.File(old_file, 'r') as of:

        current_keys = np.unique([int(key) for key in cf])
        print(f"Current file events: {current_keys.size}")
        cf.attrs['labeled_elms'] = np.array([int(elm_event_key) for elm_event_key in cf], dtype=int)
        cf.flush()
        print(f"Current file labeled: {cf.attrs['labeled_elms'].size} skipped: {cf.attrs['skipped_elms'].size}")

        old_keys = np.unique([int(key) for key in of])
        print(f"Old file events: {old_keys.size}")
        print(f"Old file labeled: {of.attrs['labeled_elms'].size} skipped: {of.attrs['skipped_elms'].size}")

        # cf_skipped_elms = np.unique(np.append(cf_skipped_elms, of_skipped_elms))
        # cf.attrs['skipped_elms'] = cf_skipped_elms
        # cf.flush()
        # print(f"Updated current file labeled: {cf_labeled_elms.size} skipped: {cf_skipped_elms.size}")

        not_present_count = 0
        for elm_event_key, elm_event in of.items():
            assert isinstance(elm_event, h5py.Group)
            assert 'labels' in elm_event
            labels = np.array(elm_event['labels'], dtype=int)
            last_inactive_elm = np.nonzero(labels==1)[0][0] - 1
            assert last_inactive_elm > 0
            if last_inactive_elm < 3400:
                continue
            print(f"ELM {elm_event_key} pre-ELM perioed: {last_inactive_elm}"
            f"\t In current file?: {elm_event_key in cf}")
            if elm_event_key not in cf:
                elm_event_id = int(elm_event_key)
                assert elm_event_id not in cf.attrs['labeled_elms']
                if elm_event_id in cf.attrs['skipped_elms']:
                    print(f"  In `skipped_elms` in current file, continuing")
                    continue
                not_present_count += 1
                if not dry_run:
                    new_group = cf.create_group(name=elm_event_key)
                    for key, value in elm_event.items():
                        new_group.create_dataset(name=key, data=value)
                    for attr_key, attr_value in elm_event.attrs.items():
                        new_group.attrs[attr_key] = attr_value
                    cf_labeled_elms = np.append(cf_labeled_elms, elm_event_id)
                    cf.attrs['labeled_elms'] = cf_labeled_elms
                    cf.flush()

        print(f"New ELM events to add: {not_present_count}")

        for elm_event_key, elm_event in cf.items():
            elm_event_id  = int(elm_event_key)
            assert elm_event_id in cf.attrs['labeled_elms']
            if elm_event_id in cf.attrs['skipped_elms']:
                print(f"Event {elm_event_key} is in `skipped_elms`")
                del cf[elm_event_key]
                assert elm_event_key not in cf
                print("  Removed")



if __name__=='__main__':
    # package_unlabeled_elm_events(max_elms=3)
    add_elm_events(dry_run=False)