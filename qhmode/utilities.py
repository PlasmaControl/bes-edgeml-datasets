from bes_data_tools.bes_data import package_bes
from bes_data_tools.package_h5 import package_8x8_signals


def get_metadata():
    package_bes(
        shotlist_csvfile='lh_shotlist.csv',
        max_shots=2,
        output_h5file='lh_metadata.hdf5',
        verbose=True,
    )

    package_bes(
        shotlist_csvfile='qh_shotlist.csv',
        max_shots=2,
        output_h5file='qh_metadata.hdf5',
        verbose=True,
    )


def get_signals():
    package_8x8_signals(
        input_h5file='lh_metadata.hdf5',
        output_h5file='lh_metadata_8x8.hdf5',
        max_shots=2,
    )

    package_8x8_signals(
        input_h5file='qh_metadata.hdf5',
        output_h5file='qh_metadata_8x8.hdf5',
        max_shots=2,
    )


if __name__=='__main__':
    pass