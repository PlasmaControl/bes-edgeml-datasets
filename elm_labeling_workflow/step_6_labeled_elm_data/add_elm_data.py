from pathlib import Path
# import shutil

from elm_labeling_workflow.elm_data_tools import HDF5_Data, print_hdf5_contents


bes_data_file = Path('/home/smithdr/ml/elm_data/step_6_labeled_elm_data/elm_data_v1.hdf5')
label_data_file = Path('/home/smithdr/ml/elm_data/step_5_label_elms/elm_label_data_v3.hdf5')
df = HDF5_Data(hdf5_file=bes_data_file)
df.update_bes_data_file(
    label_data_file=label_data_file,
    use_concurrent=True, 
    dry_run=False,
)
print("/n/n/n/n")
print_hdf5_contents(bes_data_file)

# # if bes_data_file.exists():
# #     datetime_str = time.strftime("%Y%m%d-%H%M%S")
# #     new_file = Path(f'./elm_data_v1.{datetime_str}.hdf5').absolute()
# #     shutil.copy2(src=bes_data_file, dst=new_file)

# # shutil.copy2(src=labeled_elms_file, dst=full_data_file)
# # assert full_data_file.exists()

# df = HDF5_Data(hdf5_file=bes_data_file)
# df.update_bes_data_file(
#     label_data_file=label_data_file,
#     use_concurrent=True, 
#     bes_channels='all',
#     dry_run=True,
# )
