from pathlib import Path
import csv
import re


def read_elm_shotlist(verbose=False):
    rx = re.compile(r"^\s+(?P<runid>\w+)\s+(?P<shot>\d+)\s*$")
    file = Path.home() / 'edge-ml/data/elm-shotlist.csv'
    with file.open() as f:
        reader = csv.reader(f)
        shotlist = []
        runids = []
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

if __name__=='__main__':
    runids, shotlist = read_elm_shotlist()