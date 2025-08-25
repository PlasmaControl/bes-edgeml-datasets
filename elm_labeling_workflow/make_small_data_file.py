from pathlib import Path
import numpy as np
import h5py


def make_small_data_file(
        new_file: str|Path,
        existing_file: str|Path = '/home/smithdr/ml/elm_labeling_workflow/step_6_labeled_elm_data/elm_data_v1.hdf5',
        n_elms: int = 20,
        max_elms_per_shot: int = 1,
):
    new_file = Path(new_file).absolute()
    existing_file = Path(existing_file).absolute()
    assert existing_file.exists()
    print(f"New file: {new_file}")
    print(f"  Requested ELMs: {n_elms}")
    with (
        h5py.File(existing_file, 'r') as src,
        h5py.File(new_file, 'w') as dest,
    ):
        existing_elms = [int(elm_key) for elm_key in src['elms']]
        np.random.default_rng().shuffle(existing_elms)
        print(f"  Existing ELMs: {len(existing_elms)}")

        dest.require_group('shots')
        dest.require_group('elms')
        dest_elms_per_shot: dict[int,int] = {}

        print(f"  Max ELMs per shot: {max_elms_per_shot}")
        for elm in existing_elms:
            elm_key = f"{elm:06d}"
            shot = src['elms'][elm_key].attrs['shot']
            # check for max elms per shot
            if shot in dest_elms_per_shot and dest_elms_per_shot[shot] > max_elms_per_shot:
                continue
            # check for pre-ELM time
            t_start = src['elms'][elm_key].attrs['t_start']
            t_stop = src['elms'][elm_key].attrs['t_stop']
            if t_stop-t_start < 25 or t_stop-t_start > 60:
                continue
            # increment ELM count for this shot
            if shot in dest_elms_per_shot:
                dest_elms_per_shot[shot] += 1
            else:
                dest_elms_per_shot[shot] = 1
            # copy ELM
            src.copy(
                source=src['elms'][elm_key],
                dest=dest['elms'],
            )
            # copy shot
            if str(shot) not in dest['shots']:
                src.copy(
                    source=src['shots'][str(shot)],
                    dest=dest['shots'],
                )
            if len(dest['elms']) >= n_elms:
                break
            

if __name__=='__main__':
    for n_elms in [20, 50, 100]:
        make_small_data_file(
            new_file=f'small_data_{n_elms:d}.hdf5',
            existing_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
            n_elms=n_elms,
            max_elms_per_shot=1,
        )
    n_elms = 200
    make_small_data_file(
        new_file=f'small_data_{n_elms:d}.hdf5',
        existing_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        n_elms=n_elms,
        max_elms_per_shot=2,
    )
    n_elms = 500
    make_small_data_file(
        new_file=f'small_data_{n_elms:d}.hdf5',
        existing_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        n_elms=n_elms,
        max_elms_per_shot=4,
    )