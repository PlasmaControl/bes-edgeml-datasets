from pathlib import Path
import mirpyidl


def restore_db():
    db_file = Path('data/db.idl')
    assert(db_file.exists())
    idl_command = f"RESTORE, '{db_file.as_posix()}', /VERBOSE"
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

    return heidbrink_db


if __name__ == '__main__':
    db = restore_db()
