from pathlib import Path
import csv
import re
import numpy as np
from edgeml import bes2hdf5


def read_elm_shotlist(verbose=False):
    file = Path(__file__).parent / 'elm-shotlist.csv'
    rx = re.compile(r"^\s+(?P<runid>\w+)\s+(?P<shot>\d+)\s*$")
    shotlist = []
    runids = []
    with file.open() as f:
        reader = csv.reader(f)
        for irow, row in enumerate(reader):
            if irow==0:
                continue
            match = rx.match(row[0])
            if verbose:
                print(irow, row, match)
            assert(match)
            runids.append(match.group('runid'))
            shotlist.append(eval(match.group('shot')))
    return runids, shotlist


def package_elms(nshots=None):
    _, shots = read_elm_shotlist()
    shots = np.array(shots)
    if nshots:
        shots = shots[0:nshots]
    print(shots[0], shots[-1], shots.size)
    bes2hdf5.package_bes(shots=shots, verbose=True, with_signals=False)


if __name__=='__main__':
    runids, shotlist = read_elm_shotlist()