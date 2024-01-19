from pathlib import Path
import shutil

from elm_data.data_tools import HDF5_Data, print_hdf5_contents

full_data_file = Path('./elm_data_v1.hdf5').absolute()

if not full_data_file.exists():
    labeled_elms_file = Path('/home/smithdr/ml/elm_data/step_5_label_elms/labeled_elms_v3.hdf5')
    shutil.copy2(src=labeled_elms_file, dst=full_data_file)
    assert full_data_file.exists()

df = HDF5_Data(hdf5_file=full_data_file)

df.add_bes_elm_event_data(use_concurrent=True, max_shots=None, bes_channels='all')

print_hdf5_contents(full_data_file)
