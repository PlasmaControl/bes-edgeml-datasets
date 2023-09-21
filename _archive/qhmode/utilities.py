from bes_data_tools import package_h5


def get_metadata():
    package_h5.package_bes(
        input_csvfile='lh_shotlist.csv',
        max_shots=4,
        output_hdf5file='sample_lh_metadata.hdf5',
        verbose=True,
    )

    package_h5.package_bes(
        input_csvfile='qh_shotlist.csv',
        max_shots=4,
        output_hdf5file='sample_qh_metadata.hdf5',
        verbose=True,
    )


def get_signals():
    package_h5.package_8x8_signals(
        input_hdf5file='sample_lh_metadata.hdf5',
        output_hdf5file='sample_lh_metadata_8x8.hdf5',
        max_shots=2,
    )

    package_h5.package_8x8_signals(
        input_hdf5file='sample_qh_metadata.hdf5',
        output_hdf5file='sample_qh_metadata_8x8.hdf5',
        max_shots=2,
    )


if __name__=='__main__':
    # get_metadata()
    # get_signals()
    package_h5.read_metadata('qh_metadata_8x8.hdf5')