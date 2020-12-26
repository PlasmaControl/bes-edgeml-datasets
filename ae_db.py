import mirpyidl


def restore_db():
    mirpyidl.execute("restore, '~/edge-ml/data/db.idl', /verbose")
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
