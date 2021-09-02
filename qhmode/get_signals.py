from bes_data_tools.package_h5 import package_8x8_signals

package_8x8_signals(
    input_h5file='lh_metadata.hdf5',
    output_h5file='lh_metadata_8x8.hdf5',
    max_shots=None,
)

package_8x8_signals(
    input_h5file='qh_metadata.hdf5',
    output_h5file='qh_metadata_8x8.hdf5',
    max_shots=None,
)