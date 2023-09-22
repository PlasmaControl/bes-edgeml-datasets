import dataclasses
from pathlib import Path
import csv

import numpy as np
# import MDSplus
import h5py

from bes_data_tools.bes_data_tools import BES_H5_Data, BES_Shot_Data


@dataclasses.dataclass
class Elm_List:
    input_elm_list_csv: str|Path = None  # CSV file path
    input_metadata_hdf5: str|Path = None  # HDF5 file path

    def __post_init__(self) -> None:
        self.input_elm_list_csv = Path(self.input_elm_list_csv)
        self.input_metadata_hdf5 = Path(self.input_metadata_hdf5)

    def filter_elm_list_and_get_signals(
            self,
            output_csv_file: str|Path = 'filtered_elm_list.csv',
            output_hdf5_file: str|Path = 'elm_data.hdf5',
            max_elms: int = None,
            max_elms_per_shot: int = None,
            max_shots: int = None,
            min_pinj_15l: float = 700e3,
            max_pinj_15r: float = 400e3,
    ) -> None:
        output_csv_file = Path(output_csv_file)
        output_hdf5_file = Path(output_hdf5_file)
        print(f"Input ELM list CSV file: {self.input_elm_list_csv}")
        print(f"Input metadata HDF5 file: {self.input_metadata_hdf5}")
        print(f"Output CSV file: {output_csv_file}")
        print(f"Output signal data HDF5 file: {output_hdf5_file}")
        with (
            self.input_elm_list_csv.open('r') as input_csv, 
            h5py.File(self.input_metadata_hdf5, 'r') as input_hdf5,
            output_csv_file.open('w') as output_csv,
            h5py.File(output_hdf5_file, 'w') as output_hdf5,
        ):
            csv_reader = csv.DictReader(input_csv)
            fieldnames = csv_reader.fieldnames
            assert set(['shot','start_time','stop_time']) == set(fieldnames)
            csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
            csv_writer.writeheader()
            print(f"Groups in input HDF5 file: {len(input_hdf5)}")
            output_hdf5.attrs['n_shots'] = 0
            i_elms = 0
            old_shot = -1
            skipped_elms_pinj = 0
            skipped_elms_max_signal = 0
            i_shots = 0
            for row in csv_reader:
                shot = int(row['shot'])
                shot_metadata = None
                if shot != old_shot:
                    # fetch data
                    i_elms_for_shot = 0
                    print(f'Fetching metadata for shot {shot}')
                    shot_metadata = input_hdf5[f'{shot}']
                else:
                    # reuse data
                    shot_metadata = old_shot_metadata
                assert shot_metadata is not None
                old_shot = shot
                old_shot_metadata = shot_metadata
                t_start = float(row['start_time'])
                t_stop = float(row['stop_time'])
                pinj_time = np.array(shot_metadata['pinj_time'])
                pinj_time_mask = (pinj_time >= t_start) & (pinj_time <= t_stop)
                pinj_15l = np.array(shot_metadata['pinj_15l'])[pinj_time_mask]
                pinj_15r = np.array(shot_metadata['pinj_15r'])[pinj_time_mask]
                pinj_violation = (
                    (min_pinj_15l and np.any(pinj_15l<min_pinj_15l)) or
                    (max_pinj_15r and np.any(pinj_15r>max_pinj_15r))
                )
                if pinj_violation:
                    skipped_elms_pinj += 1
                    continue
                if max_elms_per_shot and i_elms_for_shot >= max_elms_per_shot:
                    continue
                # copy metadata and get signals
                shot_group_name = str(shot)
                if shot_metadata.name not in output_hdf5:
                    if max_shots and i_shots >= max_shots:
                        break
                    i_shots += 1
                    output_hdf5.attrs['n_shots'] = i_shots
                    bes_data = BES_Shot_Data(shot=shot, skip_metadata=True)
                    assert bes_data.time is not None
                    bes_data.get_bes_signals(channels=[19,21,23])
                    shot_metadata.copy(
                        source=shot_metadata,
                        dest=output_hdf5,
                        name=shot_group_name,
                    )
                    shot_group = output_hdf5[shot_group_name]
                    shot_group.attrs['n_elms'] = 0
                    for dataset in ['signals','time']:
                        data = getattr(bes_data, dataset)
                        data = data[..., ::5]
                        shot_group.require_dataset(
                            name='bes_'+dataset,
                            data=data,
                            shape=data.shape,
                            dtype=data.dtype,
                        )
                    node_names = ['ip', 'echpwr', 'bdotampl']
                    for node_name in node_names:
                        print(f'  Getting node `{node_name}`')
                        node_data = bes_data.get_signal(
                            node_name=node_name, 
                            max_sample_rate=10e3 if 'bdot' not in node_name else 100e3,
                        )
                        for tag in ['data','time']:
                            shot_group.require_dataset(
                                name=f'{node_name}_{tag}',
                                data=node_data[tag],
                                shape=node_data[tag].shape,
                                dtype=node_data[tag].dtype,
                            )
                    node_names = ['FS03', 'FS04', 'FS05']
                    for i_node, node_name in enumerate(node_names):
                        print(f'  Getting node `{node_name}`')
                        signal_dict = bes_data.get_signal(
                            node_name=node_name,
                            tree='spectroscopy',
                            max_sample_rate=50e3,
                        )
                        shot_group.require_dataset(
                            name=node_name,
                            data=signal_dict['data'],
                            shape=signal_dict['data'].shape,
                            dtype=signal_dict['data'].dtype,
                        )
                        if i_node == 0:
                            shot_group.require_dataset(
                                name='FS_time',
                                data=signal_dict['time'],
                                shape=signal_dict['time'].shape,
                                dtype=signal_dict['time'].dtype,
                            )
                    node_names = ['denv2f', 'denv3f']
                    for i_node, node_name in enumerate(node_names):
                        print(f'  Getting node `{node_name}`')
                        signal_dict = bes_data.get_signal(
                            node_name=node_name,
                            tree='electrons',
                            max_sample_rate=100e3,
                        )
                        shot_group.require_dataset(
                            name=node_name,
                            data=signal_dict['data'],
                            shape=signal_dict['data'].shape,
                            dtype=signal_dict['data'].dtype,
                        )
                        if i_node == 0:
                            shot_group.require_dataset(
                                name='denv_time',
                                data=signal_dict['time'],
                                shape=signal_dict['time'].shape,
                                dtype=signal_dict['time'].dtype,
                            )
                    output_hdf5.flush()
                # test for valid BES signals during ELM event
                signals = np.array(output_hdf5[shot_group_name]['bes_signals'])
                time = np.array(output_hdf5[shot_group_name]['bes_time'])
                time_mask = (time >= t_start-5) & (time <= t_stop+5)
                signals = signals[:, time_mask]
                max_signals = np.max(signals, axis=1)
                if np.all(max_signals < 8):
                    skipped_elms_max_signal += 1
                    continue
                # successful ELM candidate
                i_elms += 1
                output_hdf5.attrs['n_elms'] = i_elms
                i_elms_for_shot += 1
                output_hdf5[shot_group_name].attrs['n_elms'] = i_elms_for_shot
                csv_writer.writerow(row)
                print(f"  {i_elms} {i_elms_for_shot} {shot} {t_start:.2f}")
                if i_elms%10000 == 0:
                    print(f"  i_elms {i_elms:,d} from {i_shots:,d} shots")
                if max_elms and i_elms >= max_elms:
                    break
            for shot_group_name in output_hdf5:
                if output_hdf5[shot_group_name].attrs['n_elms'] == 0:
                    output_hdf5.attrs['n_shots'] -= 1
                    del output_hdf5[shot_group_name]
        print(f"Final: i_elms {i_elms:,d} from {i_shots:,d} shots")
        print(f"  Skipped for pinj: {skipped_elms_pinj:,d}")
        print(f"  Skipped for max signal: {skipped_elms_max_signal:,d}")
        data_file = BES_H5_Data(hdf5_file=output_hdf5_file)
        data_file.print_hdf5_contents()


if __name__=='__main__':
    elms = Elm_List(
        input_elm_list_csv='/home/smithdr/ml/elm/step_4_elm_list_v2.csv',
        input_metadata_hdf5='/home/smithdr/ml/elm/data/big_metadata_v4.hdf5',
    )
    elms.filter_elm_list_and_get_signals(
        max_shots=25,
    )