from elm_data.data_tools import HDF5_Data

dataset = HDF5_Data(
    hdf5_file='data_v6.hdf5',
    truncate_hdf5=True,
    really_truncate_hdf5=True,
)
dataset.create_metadata_file(
    csv_file='/home/smithdr/ml/elm_data/step_3_filter_metadata/filtered_v2_shotlist.csv',
    bes_channels=[21,23],
    max_bes_sample_rate=200e3,
    with_limited_signals=True,
    use_concurrent=True, 
    # max_shots=32,
)
dataset.print_hdf5_contents(print_attributes=False, print_datasets=False)

dataset.plot_8x8_rz_avg()
dataset.plot_ip_bt_histograms()
dataset.plot_configurations()
