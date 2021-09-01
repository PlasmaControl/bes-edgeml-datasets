from bes_data_tools import package

package.package_signals_8x8_only(
        input_h5file='lh_metadata.hdf5',
        output_h5file='lh_metadata_8x8.hdf5',
        max_shots=2,
)

package.package_signals_8x8_only(
        input_h5file='qh_metadata.hdf5',
        output_h5file='qh_metadata_8x8.hdf5',
        max_shots=2,
)