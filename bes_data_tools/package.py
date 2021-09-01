from pathlib import Path
import csv
import time as timelib

import numpy as np
import matplotlib.pyplot as plt
import h5py

try:
    from . import bes2hdf5
except:
    import bes2hdf5


# def package_metadata(shotlist_csvfile='shotlist.csv',
#                      max_shots=None,
#                      output_h5file='bes_metadata.hdf5'):
#     shotlist_csvfile = Path(shotlist_csvfile)
#     print(f'Using shotlist {shotlist_csvfile.as_posix()}')
#     assert(shotlist_csvfile.exists())
#     shotlist = []
#     with shotlist_csvfile.open() as csvfile:
#         reader = csv.DictReader(csvfile,
#                                 fieldnames=None,
#                                 skipinitialspace=True)
#         for irow, row in enumerate(reader):
#             shotlist.append(int(row['shot']))
#             if max_shots and irow > max_shots:
#                 break
#     # filename = Path('bes_metadata.hdf5')
#     bes2hdf5.package_bes(shotlist=shotlist,
#                          verbose=True,
#                          with_signals=False,
#                          output_h5file=output_h5file,
#                          )


def make_8x8_sublist(input_h5file='metadata.hdf5',
                     upper_inboard_channel=None,
                     verbose=False,
                     noplot=False,
                     rminmax=(223,227),
                     zminmax=(-1.5,1)):
    input_h5file = Path(input_h5file)
    r = []
    z = []
    nshots = []
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
        plt.xlim(220, 230)
        plt.ylim(-1.5, 1.5)
        for r in rminmax:
            plt.vlines(r, zminmax[0], zminmax[1], color='k')
        for z in zminmax:
            plt.hlines(z, rminmax[0], rminmax[1], color='k')
        plt.xlabel('R (cm)')
        plt.ylabel('Z (cm)')
        plt.title('R/Z centers of BES 8x8 grids, and shot counts')
    return shotlist


def package_signals_8x8_only(input_h5file='metadata.hdf5',
                             max_shots=None,
                             output_h5file='metadata_8x8.hdf5'):
    input_h5file = Path(input_h5file)
    print(f'Using metadata in {input_h5file.as_posix()}')
    assert(input_h5file.exists())
    shot_list = make_8x8_sublist(
            input_h5file=input_h5file,
            upper_inboard_channel=56,
            noplot=True)
    if max_shots:
        shot_list = shot_list[0:max_shots]
    bes2hdf5.package_bes(shotlist=shot_list,
                         output_h5file=output_h5file,
                         verbose=True,
                         with_signals=True,
                         )


def package_unlabeled_elm_events(elm_csvfile=None,
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
    unlabeled_elm_events_file = Path('elm_events.hdf5')  # local output file
    signals_directory = 'signals_8x8_only'
    signals_directory.mkdir(exist_ok=True)
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
    bes2hdf5.print_h5py_contents(unlabeled_elm_events_file)
    print('Failed shots:', failed_shots)
    t2 = timelib.time()
    dt = t2 - t1
    print(f'Elapsed time: {int(dt) // 3600} hr {dt % 3600 / 60:.1f} min')


if __name__ == '__main__':
    pass