from pathlib import Path

import h5py

from edgeml.bes2hdf5 import traverse_h5py


def combine_labeled_data_files():
    labeled_data_dir = Path('/fusion/projects/diagnostics/bes/smithdr/labeled-elms/data')
    for h5_file in labeled_data_dir.glob('*/labeled-elm-events*.hdf5'):
        print(h5_file)
        traverse_h5py(h5_file, skip_subgroups=True)

if __name__ == '__main__':
    combine_labeled_data_files()