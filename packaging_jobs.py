#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BES data packaging jobs, likely to be run in Slurm scripts
"""

from pathlib import Path
import pickle
import numpy as np
from bes2hdf5 import package_bes


def package_ae_database():
    pickle_file = Path.home() / 'edge-ml/data/db.pickle'
    with pickle_file.open('rb') as f:
        ae_db = pickle.load(f)
    shots = np.unique(ae_db['shot'])
    print(shots[0], shots[-1], shots.size)
    package_bes(shots=shots, verbose=True, with_signals=False)
