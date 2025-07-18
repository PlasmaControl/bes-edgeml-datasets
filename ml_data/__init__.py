import sys as _sys
from pathlib import Path as _Path

_dirpath = _Path(__file__).parent
_hdf5_files = sorted(_dirpath.glob('**/*.hdf5'))
_hdf5_files = [_dirpath / relpath for relpath in _hdf5_files]

# print(__file__)
# print(_dirpath)
# print(_hdf5_files)
# print(_sys.modules[__name__])

for _file in _hdf5_files:
    setattr(_sys.modules[__name__], _file.stem, _file)
