import numpy as np
from elm_data_tools.elm_data_tools import HDF5_Data

dataset = HDF5_Data(
    hdf5_file='/home/smithdr/ml/elm_data/step_2_shot_metadata/metadata_v8.hdf5',
)
# dataset.print_hdf5_contents(print_attributes=False, print_datasets=False)

filtered_shotlist = dataset.filter_shots(
    filename_prefix='filtered_v2',
    r_avg=np.array([223.0, 228.0]),
    z_avg=[-1.5, 1],
    delz_avg=[0, 1.7],
    ip_extremum=[0.8e6, 2e6],
    bt_extremum=[-3, -1.8],
    pinj_15l_max=[0.7e6, 2.5e6],
    only_standard_8x8=True,
)
dataset.plot_8x8_rz_avg(shotlist=filtered_shotlist)
dataset.plot_ip_bt_histograms(shotlist=filtered_shotlist)
