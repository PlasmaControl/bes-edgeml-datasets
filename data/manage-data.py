from pathlib import Path
import numpy as np
import h5py


def ensure_unique(array):
    if isinstance(array[0], str):
        array = [int(i) for i in array]
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array = np.sort(array)
    return np.array_equal(array, np.unique(array))


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


data_dir = Path(__file__).parent

original_data_dir = data_dir / 'original_data'
assert(original_data_dir.exists())

print('Original files')
original_data_files = list(original_data_dir.glob('*.hdf5'))
total_elm_count = 0
for file in original_data_files:
    print(f'  {file.name}   size {file.stat().st_size/(1024**2):.1f} MB')
    with h5py.File(file, 'r') as h5file:
        for key, value in h5file.attrs.items():
            print(f'    {key} shape {value.size}')
        labeled_elms = h5file.attrs['labeled_elms']
        print(labeled_elms.size, len(h5file))
        # assert(ensure_unique(labeled_elms))
        # assert(labeled_elms.size == len(h5file))
        total_elm_count += len(h5file)

print(f'Total ELM count: {total_elm_count}')


if __name__=='__main__':
    combine_datafiles(files=original_data_files)