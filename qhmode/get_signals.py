from bes_data_tools.package import package_signals_8x8_only

package_signals_8x8_only(
    input_h5file='lh_metadata.hdf5',
    output_h5file='lh_metadata_8x8.hdf5',
    max_shots=None,
)

package_signals_8x8_only(
    input_h5file='qh_metadata.hdf5',
    output_h5file='qh_metadata_8x8.hdf5',
    max_shots=None,
)