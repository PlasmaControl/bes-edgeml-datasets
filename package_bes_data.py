import h5py
import numpy as np
import matplotlib.pyplot as plt
import MDSplus

connection = MDSplus.Connection('atlas.gat.com')
channels = np.arange(1,65)

def get_bes(shot=176778):
    tdi_vars = []
    tdi_assignments = []
    for channel in channels:
        var = f'_n{channel:02d}'
        tdi_vars.append(var)
        tdi_assignments.append(var+f' = ptdata("besfu{channel:02d}", {shot})')
    connection.get(', '.join(tdi_assignments))
    size = connection.get(f'size({tdi_vars[0]})')
    data = np.empty([channels.size, size.data()])
    for i,tdi_var in enumerate(tdi_vars):
        data[i,:] = connection.get(tdi_var)
    return data

shotlist = [176778, 171472, 171473, 171477, 171495,
            145747, 145745, 142300, 142294, 145384]

def package_shots(shots=shotlist):
    if not isinstance(shots, (list, np.ndarray)):
        shots = [shots]
    with h5py.File('bes_data.hdf5', 'w') as f:
        for shot in shots:
            print(f'Packaging shot {shot}')
            shot_group = f.create_group(f's{shot}')
            shot_group.create_dataset('besdata',
                                      data=get_bes(shot=shot),
                                      compression='gzip',
                                      compression_opts=7)

if __name__=='__main__':
    plt.close('all')
    package_shots()
#    data = get_bes()