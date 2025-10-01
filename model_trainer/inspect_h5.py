from pathlib import Path
from textwrap import indent
from typing import Sequence
import os
from datetime import datetime

import numpy as np
import h5py

def print_hdf5_contents(
        hdf5_file: Path|str = None,
        hdf5_group: h5py.Group = None,
        print_attributes: bool = True,
        print_datasets: bool = True,
        max_groups: int = None,
):

    def _print_attributes(obj: h5py.Group|h5py.Dataset):
        more_indent = '  ' if isinstance(obj, h5py.Dataset) else ''
        for key in obj.attrs:
            item = obj.attrs[key]
            if isinstance(item, np.ndarray):
                print(more_indent + f'          Attribute {key}: shape {item.shape} dtype {item.dtype}')
            elif isinstance(item, str):
                print(more_indent + f'          Attribute {key}: {item}')
            elif isinstance(item, Sequence):
                print(more_indent + f'          Attribute {key}: len {len(item)}')
            else:
                print(more_indent + f'          Attribute {key}: value {item} type {type(item)}')

    def _recursively_print_content(group: h5py.Group, level: int = 0):
        n_subgroups = 0
        n_datasets = 0
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                n_subgroups += 1
            elif isinstance(item, h5py.Dataset):
                n_datasets += 1
            else:
                raise ValueError
        indentation = '  ' * level
        print(f'{indentation}Group {group.name}: {n_datasets} datasets, {n_subgroups} subgroups, and {len(group.attrs)} attributes')
        if print_attributes: _print_attributes(group)
        n_groups = 0
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                n_groups += 1
                if (max_groups and n_groups <= max_groups) or not max_groups:
                    _recursively_print_content(item, level=level+1)
            elif isinstance(item, h5py.Dataset):
                if print_datasets: print(f'{indentation}  Dataset {key}: {item.shape} {item.dtype} {item.nbytes/1024/1024:,.2f} MB')
                if print_attributes: _print_attributes(item)
            else:
                raise ValueError

    if hdf5_file:
        hdf5_file = Path(hdf5_file)
        print(f'Contents of {hdf5_file}')
        assert hdf5_file.exists(), f'HDF5 file does not exist: {hdf5_file}'
        file_stat = os.stat(hdf5_file)
        size_MB = file_stat.st_size/1024/1024
        timestamp = datetime.fromtimestamp(file_stat.st_mtime).isoformat(' ', 'seconds')
        print(f'File size and last modification: {size_MB:,.1f} MB  {timestamp}')
        with h5py.File(hdf5_file, 'r') as root:
            _recursively_print_content(root, level=0)
    if hdf5_group:
        print(f'Contents of {hdf5_group.name} from file {hdf5_group.file}')
        _recursively_print_content(hdf5_group, level=0)


if __name__ == '__main__':
    file = '/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5'
    print_hdf5_contents(
        hdf5_file=file,
        print_attributes=False,
        # print_datasets=False,
        max_groups=5,
    )
