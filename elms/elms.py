from pathlib import Path
import csv
import numpy as np
from edgeml import bes2hdf5


def read_elm_shotlist(filename=None):
    if not filename:
        filename = Path(__file__).parent / 'elm-shotlist.csv'
    filename = Path(filename)
    assert(filename.exists())
    shotlist = []
    with filename.open() as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ')
        for row in reader:
            shotlist.append(int(row['shot']))
    return np.array(shotlist)


def package_shotlist_metadata(max_shots=None):
    shotlist = read_elm_shotlist()
    if max_shots:
        shotlist = shotlist[0:max_shots]
    print(shotlist[0], shotlist[-1], shotlist.size)
    bes2hdf5.package_bes(shots=shotlist,
                         filename='shotlist_metadata.hdf5',
                         verbose=True,
                         with_signals=False,
                         max_workers=2)


def package_8x8_sublist(max_shots=None):
    shotlist = bes2hdf5.make_8x8_sublist(
            path='/home/smithdr/edgeml/elms/data/elm_metadata/bes_metadata.hdf5',
            upper_inboard_channel=56,
            noplot=True)
    if max_shots:
        shotlist = shotlist[0:max_shots]
    bes2hdf5.package_bes(shots=shotlist,
                         channels=np.arange(1,65),
                         verbose=True,
                         with_signals=True)


if __name__=='__main__':
    # shotlist = read_elm_shotlist()
    package_shotlist_metadata(max_shots=4)
    # bes2hdf5.print_metadata_summary('data/elm_metadata/bes_metadata.hdf5', only_8x8=True)
    # shotlist = bes2hdf5.make_8x8_sublist(path='data/elm_metadata/bes_metadata.hdf5', upper_inboard_channel=56)
    # package_8x8_sublist()
