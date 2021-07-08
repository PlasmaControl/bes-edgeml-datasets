from pathlib import Path
import datetime

import numpy as np
import h5py

from edgeml.bes2hdf5 import traverse_h5py


data_dir = Path('data').absolute()
data_dir.mkdir(exist_ok=True)

figure_dir = Path('figures').absolute()
figure_dir.mkdir(exist_ok=True)


def ensure_unique(array):
    if isinstance(array[0], str):
        array = [int(i) for i in array]
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array = np.sort(array)
    return np.array_equal(array, np.unique(array))


def combine_labeled_data_files():
    # original labeled data
    original_data_dir = Path('/fusion/projects/diagnostics/bes/smithdr/labeled-elms/data')
    original_data_files = list(original_data_dir.glob('*/labeled-elm-events*.hdf5'))
    # new combined data file
    combined_data_file = data_dir / 'labeled-elm-events.hdf5'
    # rename if exists
    if combined_data_file.exists():
        new_filename = f"labeled-elm-events-{datetime.strftime('%Y-%m-%d-%H-%M-%S')}.hdf5"
        print(f'Renaming old data file: {new_filename}')
        combined_data_file.rename(data_dir / new_filename)
    assert(combined_data_file.exists() is False)
    # create new combined data file
    with h5py.File(combined_data_file, 'w') as combined_data:
        for original_data_file in original_data_files:
            print(original_data_file)
            traverse_h5py(original_data_file, skip_subgroups=True)
            with h5py.File(original_data_file, 'r') as original_data:
                print(original_data.attrs['labeled_elms'].size, len(original_data))
                # assert(original_data.attrs['labeled_elms'].size == len(original_data))
                # assert(ensure_unique(original_data.attrs['labeled_elms']))


def combine_datafiles(
        files=(),
        newfile='labeled-elm-events.hdf5',
        ):
    with h5py.File(newfile, 'w') as nf:
        for i, filename in enumerate(files):
            with h5py.File(filename, 'r') as f:
                for attrname in f.attrs:
                    assert(attrname in ['labeled_elms', 'skipped_elms'])
                    if i == 0:
                        nf.attrs.create(attrname, f.attrs[attrname])
                    else:
                        nf.attrs[attrname] = np.append(nf.attrs[attrname], f.attrs[attrname])
                for elm_key, elm_group in f.items():
                    assert(isinstance(elm_group, h5py.Group))
                    new_elm_group = nf.create_group(elm_key)
                    for ds_key, ds_value in elm_group.items():
                        assert(isinstance(ds_value, h5py.Dataset))
                        new_elm_group.create_dataset(ds_key, data=ds_value)
                nf.attrs['labeled_elms'] = np.unique(nf.attrs['labeled_elms'])
        labeled_elms = nf.attrs['labeled_elms']
        print(f'Size of `labeled_elms` array: {labeled_elms.size}')
        print(f'Number of groups: {len(nf)}')
        assert (ensure_unique(labeled_elms))
        assert (labeled_elms.size == len(nf))



# original_data_dir = data_dir / 'original_data'
# assert(original_data_dir.exists())
#
# print('Original files')
# original_data_files = list(original_data_dir.glob('*.hdf5'))
# total_elm_count = 0
# for file in original_data_files:
#     print(f'  {file.name}   size {file.stat().st_size/(1024**2):.1f} MB')
#     with h5py.File(file, 'r') as h5file:
#         for key, value in h5file.attrs.items():
#             print(f'    {key} shape {value.size}')
#         labeled_elms = h5file.attrs['labeled_elms']
#         print(labeled_elms.size, len(h5file))
#         # assert(ensure_unique(labeled_elms))
#         # assert(labeled_elms.size == len(h5file))
#         total_elm_count += len(h5file)
#
# print(f'Total ELM count: {total_elm_count}')


if __name__=='__main__':
    combine_labeled_data_files()