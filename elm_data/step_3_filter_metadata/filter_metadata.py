import numpy as np
from elm_data.data_tools import HDF5_Data

dataset = HDF5_Data(
    # hdf5_file='/home/smithdr/ml/elm_data/step_2_shot_metadata/metadata_v6.hdf5',
    hdf5_file='/home/smithdr/ml/elm_data/step_2_shot_metadata/metadata_v6.hdf5',
)
dataset.print_hdf5_summary()

filtered_shotlist = dataset.filter_shots(
    r_avg = [223.0, 228.0],
    z_avg = [-1.5, 1],
    ip = [0.8e6, 2e6],
    bt = [-3, -1.8],
    pinj_15l = [0.7e6, 2.5e6],
)
dataset.plot_8x8_rz_avg(shotlist=filtered_shotlist)
dataset.plot_ip_bt_histograms(shotlist=filtered_shotlist)
