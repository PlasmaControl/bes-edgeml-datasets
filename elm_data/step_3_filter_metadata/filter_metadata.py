from elm_data.data_tools import HDF5_Data

dataset = HDF5_Data(
    hdf5_file='/home/smithdr/ml/elm_data/step_2_shot_metadata/metadata_v4.hdf5',
)
dataset.print_hdf5_summary()
filter_kwargs = dict(
    r_avg = [223.0, 227.5],
    z_avg = [-1.5, 1],
    ip = [0.8e6, 2e6],
    bt = [-3, -1.8],
    pinj_15l = [0.7e6, 2.5e6],
)
dataset.filter_and_export(
    filename_prefix='filtered',
    export_csv=True,
    **filter_kwargs,
)
dataset.plot_8x8_rz_avg(
    filename='filtered_config.pdf',
    **filter_kwargs,
)
dataset.plot_ip_bt_histograms(
    filename='filtered_histograms.pdf',
    **filter_kwargs,
)
