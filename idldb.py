import mirpyidl

# Restore IDL data
mirpyidl.execute("restore, 'db.idl', /verbose")

# IDL variable names from Heidbrink documentation
heidbrink_keys = [
    'SHOT',
    'TIME',
    'EAE',
    'TAE',
    'RSAE',
    'BAE',
    'BAAE',
    'EGAM',
    'EFIT',
    'ZIP',
    'NB',
    'TENERGY0',
    'TPAS0',
    'TENERGYM',
    'TPASM',
    'FTAE',
    'FRSAE',
    'FBAE',
    'FBAAE',
    'FROT',
]

# Stuff IDL variables into a python dictionary
heidbrink_db = {}
for key in heidbrink_keys:
    heidbrink_db[key] = mirpyidl.getVariable(key)