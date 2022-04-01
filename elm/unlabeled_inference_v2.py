from pathlib import Path
import pickle
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

from elm_prediction.src import utils


def restore_model(
    model_dir: Union[Path,str,None] = None,
) -> torch.nn.Module:

    # verify model source
    args_file = Path(model_dir) / 'args.pkl'
    assert args_file.exists()

    # restore arguments namespace
    with args_file.open('r') as f:
        args = pickle.load(f)

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # instantiate model
    model_cls = utils.create_model_class(args.model_name)
    model = model_cls(args)

    # load model parameters
    _, checkpoint_file = utils.create_output_paths(args)
    load_obj = torch.load(checkpoint_file.as_posix(), map_location=device)
    model_dict = load_obj['model']
    model.load_state_dict(model_dict)

    # send to device and set to eval mode
    model.to(device)
    model.eval()

    return model


if __name__=='__main__':

    labeled_data_file = Path()
    metadata_file = Path()
    raw_data_dir = Path()

    model_dir = Path()

    model = restore_model(model_dir=model_dir)

    # determine labeled ELM ids and 
    with h5py.File(labeled_data_file.as_posix(), 'r') as f:
        skipped_elms = f.attrs['skipped_elms']
        labeled_elms_attrs = f.attrs['labeled_elms']
        labeled_elm_keys = np.array([int(key) for key in f.keys()])

    assert np.array_equal(labeled_elm_keys, labeled_elms_attrs)
    labeled_elms = labeled_elm_keys

    # determine all ELM events
    with h5py.File(metadata_file.as_posix()):
        pass