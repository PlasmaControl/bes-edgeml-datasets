from pathlib import Path
import csv
import numpy as np
from edgeml import bes2hdf5

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
    print(shotlist[0], shotlist[-1], shotlist.size)
    bes2hdf5.package_bes(shots=shotlist,
                         verbose=True,
                         with_signals=False)


def package_8x8_sublist(max_shots=None, with_signals=False):
    metadata_file = directory / 'data/elm_metadata/bes_metadata.hdf5'
    shotlist = bes2hdf5.make_8x8_sublist(
            path=metadata_file,
            upper_inboard_channel=56,
            noplot=True)
    if max_shots:
        shotlist = shotlist[0:max_shots]
    bes2hdf5.package_bes(shots=shotlist,
                         verbose=True,
                         with_signals=with_signals)


if __name__=='__main__':
    # shotlist = read_shotlist()
    # package_shotlist_metadata(max_shots=4)
    # bes2hdf5.print_metadata_summary('data/elm_metadata/bes_metadata.hdf5', only_8x8=True)
    shotlist = bes2hdf5.make_8x8_sublist(path='data/elm_metadata/bes_metadata.hdf5',
                                         upper_inboard_channel=56)
    # package_8x8_sublist()
