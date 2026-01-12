from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import h5py

try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


def plot_stats(
    max_elms: int = None,
    data_file: str = None,
    figure_dir: str = '.', 
    block_show: bool = True,
    mask_sigma_outliers: float = 8.,
    max_std: float = np.inf,
    max_channels_above_sigma: int = np.inf,
    save: bool = True, 
    merge: bool = True,
    bad_elm_indices_csv: bool = True,
    bad_elm_indices: list = None,
    skip_elm_plots: bool = False,
):
    datamodule = elm_datamodule.ELM_Datamodule(
        data_file=data_file,
        max_elms=max_elms,
        mask_sigma_outliers=mask_sigma_outliers,
        bad_elm_indices_csv=bad_elm_indices_csv,
        bad_elm_indices=bad_elm_indices,
        fraction_validation=0.,
        fraction_test=1.,
        max_predict_elms=None,
    )
    
    datamodule.setup(stage='predict')
    dataloaders = datamodule.predict_dataloader()
    n_elms = len(dataloaders)
    i_page = 1
    all_channel_stats = {
        'max_std': [],
        'min_maxabs': [],
        'max_maxabs': [],
        'channels_above_sigma': [],
    }
    if skip_elm_plots:
        merge = False
    count_rejected_elms = 0
    for i_elm, dataloader in enumerate(dataloaders):
        dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
        channel_wise_stats = dataset.pre_elm_stats()
        all_channel_stats['max_std'].append(np.amax(channel_wise_stats['std']))
        all_channel_stats['min_maxabs'].append(np.amin(channel_wise_stats['maxabs']))
        all_channel_stats['max_maxabs'].append(np.amax(channel_wise_stats['maxabs']))
        all_channel_stats['channels_above_sigma'].append(
            np.count_nonzero(channel_wise_stats['maxabs']>datamodule.max_abs_valid_signal)
        )
        elm_index = dataset.elm_index
        shot = dataset.shot
        pre_elm_size = dataset.active_elm_start_index-1
        acceptable_elm = (
            True if np.all(np.array(channel_wise_stats['std']) <= max_std) 
            and all_channel_stats['channels_above_sigma'][-1] <= max_channels_above_sigma
            else False
        )
        if not acceptable_elm:
            count_rejected_elms += 1
        if skip_elm_plots:
            continue
        # plot stats
        if i_elm==0:
            _, axes = plt.subplots(ncols=5, nrows=4, figsize=(14, 8.5))
            axes = axes.flatten()
            n_elms_per_page = axes.size // 2
        i_elm_on_page = i_elm%n_elms_per_page
        if i_elm_on_page == 0:
            plt.suptitle(f"Channel-wise pre-ELM stats (page {i_page})")
            for ax in axes:
                ax.clear()
        plt.sca(axes[i_elm_on_page + 5*(i_elm_on_page//5)])
        for key in channel_wise_stats:
            plt.plot(np.arange(1,65), channel_wise_stats[key].flatten(), label=key)
        plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
        plt.axhline(datamodule.max_abs_valid_signal, linestyle='--', color='k', linewidth=0.5)
        plt.title(
            f"ELM index {elm_index} Shot {shot}", 
            fontsize='medium',
            color='k' if acceptable_elm else 'r',
            fontweight='regular' if acceptable_elm else 'bold',
        )
        plt.xlabel('Channel', fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.yticks(fontsize='medium')
        plt.ylim(-1,1.3*datamodule.max_abs_valid_signal)
        if i_elm_on_page==0:
            plt.legend(loc='lower right', fontsize='small')
        # plot time-series signals
        plt.sca(axes[i_elm_on_page + 5*(i_elm_on_page//5) + 5])
        max_abs_channel = np.unravel_index(np.argmax(channel_wise_stats['maxabs']), channel_wise_stats['maxabs'].shape)
        max_std_channel = np.unravel_index(np.argmax(channel_wise_stats['std']), channel_wise_stats['std'].shape)
        interval = np.amax([pre_elm_size//500,1])
        time_axis = (np.arange(-pre_elm_size,0)/1e3)[::interval]
        plt.plot(
            time_axis, 
            dataset.signals[0, 0:pre_elm_size:interval, max_std_channel[0], max_std_channel[1]],
            label=f"Ch. {max_std_channel[1]+1 + 8*max_std_channel[0]}",
            alpha=0.8
        )
        if not np.array_equal(max_abs_channel, max_std_channel):
            plt.plot(
                time_axis, 
                dataset.signals[0, 0:pre_elm_size:interval, max_abs_channel[0], max_abs_channel[1]],
                label=f"Ch. {max_abs_channel[1]+1 + 8*max_abs_channel[0]}",
                alpha=0.8,
            )
        plt.legend(fontsize='small')
        plt.title(
            f"ELM index {elm_index} Shot {shot}", 
            fontsize='medium',
            color='k' if acceptable_elm else 'r',
            fontweight='regular' if acceptable_elm else 'bold',
        )
        plt.xlabel('Time-to-ELM (ms)', fontsize='medium')
        plt.ylabel('Scaled BES signals', fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.yticks(fontsize='medium')
        plt.axhline(datamodule.max_abs_valid_signal, linestyle='--', color='k', linewidth=0.5)
        plt.axhline(-datamodule.max_abs_valid_signal, linestyle='--', color='k', linewidth=0.5)
        plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
        plt.ylim(np.array([-1.3,1.3])*datamodule.max_abs_valid_signal)
        if i_elm_on_page==n_elms_per_page-1 or i_elm==n_elms-1:
            plt.tight_layout()
            if save:
                filepath = os.path.join(figure_dir, f'elm_stats_{i_page:03d}.pdf')
                print(f"Saving figure {filepath}")
                plt.savefig(filepath, format='pdf', transparent=True)
            i_page += 1
            plt.show(block=block_show)

    n_elms_above_max_std = np.count_nonzero(np.array(all_channel_stats['max_std'])>max_std)
    print(f"ELMs above max_std {max_std:.3f}: {n_elms_above_max_std}")
    print(f"Rejected ELMs: {count_rejected_elms}")

    _, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Distribution of maximum channel-wise stats during pre_ELM phase')
    axes = axes.flatten()
    for i_axis, key in enumerate(all_channel_stats):
        plt.sca(axes[i_axis])
        plt.hist(all_channel_stats[key], bins=31)
        plt.ylabel('# ELMs')
        plt.yscale('log')
        plt.ylim(bottom=0.8)
        if key=='max_std' and np.isfinite(max_std):
            plt.axvline(max_std, linestyle='--', color='k', linewidth=0.5)
        plt.xlabel(key)
    plt.tight_layout()
    if save:
        filepath = os.path.join(figure_dir, f'elm_stats_summary.pdf')
        print(f"Saving figure {filepath}")
        plt.savefig(filepath, format='pdf', transparent=True)
    plt.show(block=block_show)

    if merge:
        inputs = sorted(Path(figure_dir).glob('elm_stats_*.pdf'))
        assert len(inputs) > 0 and inputs[0].exists()
        output = Path(figure_dir) / 'elm_stats.pdf'
        output.unlink(missing_ok=True)
        gs_cmd = shutil.which('gs')
        assert gs_cmd is not None, \
            "`gs` command (ghostscript) not found; available in conda-forge"
        cmd = [
            gs_cmd,
            '-q',
            '-dBATCH',
            '-dNOPAUSE',
            '-sDEVICE=pdfwrite',
            '-dPDFSETTINGS=/prepress',
            '-dCompatibilityLevel=1.4',
            f"-sOutputFile={output.as_posix()}",
        ]
        cmd.extend([f"{pdf_file.as_posix()}" for pdf_file in inputs])
        print(f"Merging files into {output}")
        result = subprocess.run(cmd, check=True)
        assert result.returncode == 0 and output.exists()
        for pdf_file in inputs:
            pdf_file.unlink(missing_ok=True)

    return


def delete_elms(
    data_file: str|Path = None,
    elm_indices: list = None,
) -> None:
    data_file = Path(data_file)
    assert data_file.exists()
    with h5py.File(data_file, 'r+') as h5_file:
        print(f"Initial ELM count in H5 file: {len(h5_file)}")
        for elm_index in elm_indices:
            elm_key = f"{elm_index:05d}"
            if elm_key in h5_file:
                print(f"  Deleting group `{elm_key}`")
                del h5_file[elm_key]
                h5_file.flush()
            else:
                print(f"  Group `{elm_key}` not in H5 file")
        print(f"Final ELM count in H5 file: {len(h5_file)}")
    return


def make_new_data_file(
    data_file: str|Path = None,
    new_file_name: str = './test_data_50.hdf5',
    n_elms: int = 50,
):
    data_file = Path(data_file)
    assert data_file.exists()
    with h5py.File(data_file, 'r+') as h5_file:
        print(f"Initial ELM count in H5 file: {len(h5_file)}")
        keys = None


if __name__=='__main__':
    # delete_elms(
    #     data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
    #     elm_indices=[77, 217, 219, 221, 223, 239, 247, 262, 315, 319],
    # )
    plot_stats(
        data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        # mask_sigma_outliers=6,
        # max_std=5.,
        # max_channels_above_sigma=18,
        # max_elms=200,
        bad_elm_indices_csv=False,
        # skip_elm_plots=True,
        # save=False,
        # merge=False,
        block_show=False,
    )
    