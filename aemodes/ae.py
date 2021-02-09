from pathlib import Path
import pickle
import mirpyidl
import numpy as np
import bes2hdf5

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

data_dir = bes2hdf5.data_dir / 'elm'
data_dir.mkdir(exist_ok=True)


def restore_db():
    db_file = Path.home() / 'edge-ml/data/db.idl'
    assert(db_file.exists())
    idl_command = f"RESTORE, '{db_file.as_posix()}'"
    print(f'Executing python command: "{idl_command}"')
    mirpyidl.execute(idl_command)
    heidbrink_keys = [
        'SHOT', 'TIME', 'EFIT', 'ZIP', 'NB',
        'EAE', 'TAE', 'RSAE', 'BAE', 'BAAE', 'EGAM',
        'FTAE', 'FRSAE', 'FBAE', 'FBAAE', 'FROT',
        'TENERGY0', 'TPAS0', 'TENERGYM', 'TPASM',
    ]

    heidbrink_db = {}
    for key in heidbrink_keys:
        heidbrink_db[key.lower()] = mirpyidl.getVariable(key)

    pickle_file = db_file.parent / 'db.pickle'
    with pickle_file.open('wb') as f:
        pickle.dump(heidbrink_db, f)

    return heidbrink_db


def package_ae_database():
    pickle_file = Path.home() / 'edge-ml/data/db.pickle'
    with pickle_file.open('rb') as f:
        ae_db = pickle.load(f)
    shots = np.unique(ae_db['shot'])
    print(shots[0], shots[-1], shots.size)
    bes2hdf5.package_bes(shots=shots, verbose=True, with_signals=False)


if __name__ == '__main__':
    db = restore_db()
