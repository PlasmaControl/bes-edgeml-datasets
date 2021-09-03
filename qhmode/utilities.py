from bes_data_tools.package_h5 import package_bes, package_8x8_signals


def get_metadata():
    package_bes(
        input_csvfile='data/lh_shotlist.csv',
        max_shots=4,
        output_hdf5file='data/sample_lh_metadata.hdf5',
        verbose=True,
    )

    package_bes(
        input_csvfile='data/qh_shotlist.csv',
        max_shots=4,
        output_hdf5file='data/sample_qh_metadata.hdf5',
        verbose=True,
    )


def get_signals():
    package_8x8_signals(
        input_hdf5file='data/sample_lh_metadata.hdf5',
        output_hdf5file='data/sample_lh_metadata_8x8.hdf5',
        max_shots=2,
    )

    package_8x8_signals(
        input_hdf5file='data/sample_qh_metadata.hdf5',
        output_hdf5file='data/sample_qh_metadata_8x8.hdf5',
        max_shots=2,
    )


if __name__=='__main__':
    get_metadata()
    get_signals()