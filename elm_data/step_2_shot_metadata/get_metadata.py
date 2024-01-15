from elm_data.data_tools import HDF5_Data

dataset = HDF5_Data(
    hdf5_file='metadata_v8.hdf5',
    truncate_hdf5=True,
    really_truncate_hdf5=True,
)

dataset.create_metadata_file(
    csv_file='/home/smithdr/ml/elm_data/step_1_shotlist_from_D3DRDB/shotlist.csv',
    # max_shots=128,
    only_8x8=True,
    only_standard_8x8 = True,
    only_pos_ip = True,
    only_neg_bt = True,
    min_pinj_15l = 500e3,  # power in W
    min_sustained_15l = 300.,  # time in ms
    use_concurrent=True, 
)

dataset.print_hdf5_contents()
dataset.plot_8x8_rz_avg()
dataset.plot_ip_bt_histograms()
dataset.plot_configurations()
