from pathlib import Path
from datetime import datetime
import subprocess, glob, shlex

import numpy as np
import matplotlib.pyplot as plt
import h5py

try:
    from ...bes2hdf5 import traverse_h5py
except:
    from edgeml.bes2hdf5 import traverse_h5py

data_dir = Path('data')
figure_dir = Path('figures')
for subdir in [data_dir, figure_dir]:
    subdir.mkdir(exist_ok=True)

# original labeled data
original_data_dir = Path(
    '/fusion/projects/diagnostics/bes/smithdr/labeled-elms/data')
assert(original_data_dir.exists())
original_data_files = list(original_data_dir.glob('*/labeled-elm-events*.hdf5'))
assert(original_data_files)

# combined data file
combined_data_file = data_dir / 'labeled-elm-events.hdf5'



def ensure_unique(array):
    if isinstance(array[0], str):
        array = [int(i) for i in array]
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array = np.sort(array)
    return np.array_equal(array, np.unique(array))


def combine_labeled_data_files():
    # rename combined data file if exists
    if combined_data_file.exists():
        new_filename = f"labeled-elm-events-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.hdf5"
        print(f'Renaming old data file: {new_filename}')
        combined_data_file.rename(data_dir / new_filename)
    assert(combined_data_file.exists() is False)
    # create new combined data file
    with h5py.File(combined_data_file, 'w') as combined_data:
        for ifile, original_data_file in enumerate(original_data_files):
            print(original_data_file)
            traverse_h5py(original_data_file, skip_subgroups=True)
            with h5py.File(original_data_file, 'r') as original_data:
                print(original_data.attrs['labeled_elms'].size, len(original_data))
                # assert(original_data.attrs['labeled_elms'].size == len(original_data))
                # assert(ensure_unique(original_data.attrs['labeled_elms']))
                for attrname in original_data.attrs:
                    assert(attrname in ['labeled_elms', 'skipped_elms'])
                    if ifile == 0:
                        combined_data.attrs.create(attrname, original_data.attrs[attrname])
                    else:
                        combined_data.attrs[attrname] = np.append(combined_data.attrs[attrname],
                                                                  original_data.attrs[attrname])
                for elm_key, elm_group in original_data.items():
                    assert(isinstance(elm_group, h5py.Group))
                    new_elm_group = combined_data.create_group(elm_key)
                    for ds_key, ds_value in elm_group.items():
                        assert(isinstance(ds_value, h5py.Dataset))
                        new_elm_group.create_dataset(ds_key, data=ds_value)
                    for attr_name, attr_value in elm_group.attrs.items():
                        new_elm_group.attrs[attr_name] = attr_value
                combined_data.attrs['labeled_elms'] = np.unique(combined_data.attrs['labeled_elms'])
        labeled_elms = combined_data.attrs['labeled_elms']
        print(f'Size of `labeled_elms` array: {labeled_elms.size}')
        print(f'Number of groups: {len(combined_data)}')
        assert (ensure_unique(labeled_elms))
        assert (labeled_elms.size == len(combined_data))
    traverse_h5py(combined_data_file, skip_subgroups=True)


def plot_pdfs(interactive=True):
    assert(combined_data_file.exists())
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    downsample = 10
    linewidth = 0.75
    plt.figure(figsize=(10, 6))
    with h5py.File(combined_data_file, 'r') as hfile:
        i_page = 0
        n_elms = len(hfile)
        elms_per_page = 8
        total_pages = n_elms // elms_per_page +1
        for i_elm, group_name in enumerate(hfile):
            i_elm_page = i_elm % elms_per_page
            if i_elm_page == 0:
                i_page += 1
                print(f'Plotting page {i_page} of {total_pages}...')
            elm_group = hfile[group_name]
            time = np.array(elm_group['time'])
            signals = np.array(elm_group['signals']) / 10
            labels = np.array(elm_group['labels'])
            ax1 = plt.subplot(4,4,2*i_elm_page+1)
            plt.plot(time[::downsample], signals[[20,22,24],::downsample].T, color='C0', lw=linewidth)
            plt.plot(time[::downsample], labels[::downsample], color='C1', lw=linewidth)
            ax2 = plt.subplot(4,4,2*i_elm_page+2)
            one_indices = np.nonzero(labels)[0]
            i1 = np.max([one_indices[0] - one_indices.size//2, 0])
            i2 = np.min([one_indices[-1] + one_indices.size//2, time.size])
            plt.plot(time[i1:i2], signals[[20,22,24],i1:i2].T, color='C0', lw=linewidth)
            plt.plot(time[i1:i2], labels[i1:i2], color='C1', lw=linewidth)
            plt.annotate('zoomed',
                         (0.78, 0.88),
                         xycoords='axes fraction',
                         fontsize='x-small',
                         color='red')
            for ax in [ax1, ax2]:
                plt.sca(ax)
                plt.xticks(size='small')
                plt.yticks(size='small')
                plt.annotate(f"Sh {elm_group.attrs['shot']}",
                             (0.03,0.88),
                             xycoords='axes fraction',
                             fontsize='x-small')
                plt.annotate(f'Idx {group_name}',
                             (0.03,0.76),
                             xycoords='axes fraction',
                             fontsize='x-small')
            if i_elm_page == elms_per_page-1 or i_elm == n_elms-1:
                plt.tight_layout(pad=0.3)
                if interactive:
                    plt.waitforbuttonpress()
                else:
                    filepath = figure_dir / f'labeled_elms_p{i_page:03d}.pdf'
                    plt.savefig(filepath,
                                format='pdf',
                                transparent=True)
                plt.clf()
    if not interactive:
        print('Combining PDF files...')
        input_pdf_list = sorted(glob.glob('figures/labeled_elms_p*.pdf'))
        output_pdf = figure_dir / 'labeled_elms.pdf'
        command = 'gs -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress ' + \
                  f'-o {output_pdf.as_posix()} ' + \
                  ' '.join(input_pdf_list)
        result = subprocess.run(shlex.split(command))
        result.check_returncode()
        print('Finished')


def plot_pdfs_2(interactive=True):
    assert(combined_data_file.exists())
    if interactive:
        plt.ion()
    else:
        plt.ioff()
    linewidth = 0.75
    downsample = 4
    plt.figure(figsize=(10,7))
    with h5py.File(combined_data_file, 'r') as hfile:
        i_page = 0
        n_elms = len(hfile)
        elms_per_page = 4
        total_pages = n_elms // elms_per_page +1
        for i_elm, group_name in enumerate(hfile):
            i_elm_page = i_elm % elms_per_page
            if i_elm_page == 0:
                i_page += 1
                print(f'Plotting page {i_page} of {total_pages}...')
            elm_group = hfile[group_name]
            time = np.array(elm_group['time'])[::downsample]
            signals = np.array(elm_group['signals'])[:, ::downsample] / 10
            labels = np.array(elm_group['labels'])[::downsample]
            assert(time.size == labels.size and time.size == signals.shape[1])
            one_indices = np.nonzero(labels)[0]
            i1 = np.max([one_indices[0] - one_indices.size, 0])
            i2 = np.min([one_indices[-1] + one_indices.size, labels.size])
            for i_plot in range(4):
                plt.subplot(4, 4, 4*i_elm_page + i_plot + 1)
                signals_16 = signals[np.arange(i_plot*16, (i_plot+1)*16), i1:i2]
                time_mod = (time[i1:i2]-time[i1]) * 1e3
                plt.plot(time_mod, signals_16.T, color='C0', lw=linewidth)
                plt.plot(time_mod, labels[i1:i2], color='C1', lw=linewidth)
                plt.xticks(size='small')
                plt.yticks(size='small')
                plt.xlabel('Time (micro-s)', size='small')
                plt.annotate(f"Sh {elm_group.attrs['shot']}",
                             (0.02,0.88),
                             xycoords='axes fraction',
                             fontsize='x-small')
                plt.annotate(f"T0 {time[0]:.2f} ms",
                             (0.02,0.76),
                             xycoords='axes fraction',
                             fontsize='x-small')
                plt.annotate(f'Idx {group_name}',
                             (0.75,0.88),
                             xycoords='axes fraction',
                             fontsize='x-small')
                plt.annotate(f'Ch {i_plot*16+1}-{(i_plot+1)*16}',
                             (0.75, 0.76),
                             xycoords='axes fraction',
                             fontsize='x-small')
            if i_elm_page == elms_per_page-1 or i_elm == n_elms-1:
                plt.tight_layout(pad=0.4)
                if interactive:
                    plt.waitforbuttonpress()
                else:
                    filepath = figure_dir / f'labeled_elms_all_chan_p{i_page:03d}.pdf'
                    plt.savefig(filepath,
                                format='pdf',
                                transparent=True)
                plt.clf()
    if not interactive:
        print('Combining PDF files...')
        input_pdf_list = sorted(glob.glob('figures/labeled_elms_all_chan_p*.pdf'))
        output_pdf = figure_dir / 'labeled_elms_all_chan.pdf'
        command = 'gs -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress ' + \
                  f'-o {output_pdf.as_posix()} ' + \
                  ' '.join(input_pdf_list)
        result = subprocess.run(shlex.split(command))
        result.check_returncode()
        print('Finished')


def remove_labeled_elms(elm_index_list=()):
    assert(isinstance(elm_index_list, (list, tuple)))
    assert(len(original_data_files) == 3)
    print(f'Trying to delete {len(elm_index_list)} labeled ELMs from HDF5 files')
    with h5py.File(original_data_files[0], 'a') as h0, \
        h5py.File(original_data_files[1], 'a') as h1, \
        h5py.File(original_data_files[2], 'a') as h2:
        for elm_index in elm_index_list:
            assert(isinstance(elm_index, int))
            elm_str = f"{elm_index:05d}"
            for i_hfile, hfile in enumerate([h0, h1, h2]):
                if elm_str in hfile:
                    print(f'Found ELM {elm_str} in file:')
                    print(f'  {hfile.filename}')
                    response = input(f"  Permanently delete ELM {elm_str}? (y/n): ")
                    if response == 'y':
                        labeled_elms = hfile.attrs['labeled_elms']
                        idx = np.nonzero(labeled_elms == elm_index)[0]
                        labeled_elms = np.delete(labeled_elms, idx)
                        hfile.attrs['labeled_elms'] = labeled_elms
                        del hfile[elm_str]
                        print(f'  ELM {elm_str} deleted')
                    else:
                        print(f'  Skipping ELM {elm_str}')
                    break
                else:
                    if i_hfile == 2:
                        print(f'  ELM {elm_str} not found!')


# candidate_bad_elms = [
#     # 1282,  # ok
#     # 1738,  # ok
#     # 1799,  # ok
#     # 1852,  # ok
#     # 1898,  # ok
#     # 1900,  # ok
#     # 2364,  # ok
#     # 3172,  # ok
#     3426,  # strange post-ELM waveforms
#     3960,  # strange waveforms
#     3556,  # all ampl. < 0.5
#     # 5226,  # ok
#     4096, 4102, 4116, 4136, 4176, 4188, 4194, 4204,
#     4358, 4588, 5212, 5276, 5480,
#     5546,  # strange, consider deleting
#     5684, 5692, 5694, 5775,
#     6130,  # strange, consider deleting
#     6150, 6446,
#     # 6566,  # ok
#     7168,  # low amplitudes, odd waveforms
#     8456,  # low ampl
#     # 8468,  # low ampl, but other ELMs in shot ok
#     # 8558,  # ok
#     # 8590,  # ok
#     8750,  # low amplt. consider deleting
#     10226, # strange waveforms
# ]

candidate_bad_elms_2 = [
    6, 12, 24, 26,
    60, 70, 72, 142,
    332,
    342, 412,
    672, 698,
    726, 802, 824, 1076, 1127,
    1222, 1310, 1306, 3615,
    4046, 4212, 4514, 4894, 5494, 5818,
    5994, 6230, 6686, 6798, 6799,
    7936, 7966, 8274, 8432, 8711, 8590,
    8906, 8934, 8950, 9575, 9838, 9894, 9898,
]

if __name__=='__main__':
    # combine_labeled_data_files()
    plt.close('all')
    plot_pdfs_2(interactive=False)
    # remove_labeled_elms(candidate_bad_elms_2)