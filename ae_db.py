from pathlib import Path
import pickle
import mirpyidl


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


if __name__ == '__main__':
    db = restore_db()
