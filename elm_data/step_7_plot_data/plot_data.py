from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_shots(
        file: str|Path = '',
        max_shots: int = None,
        merge_pdf: bool = True,
) -> None:
    file = Path(file).absolute()
    print(f"Data file: {file}")
    assert file.exists()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 7.5))

    with h5py.File(file, 'r') as root:
        shots = root['shots']
        n_shots = len(shots)
        print(f"Number of shots: {n_shots}")
        for i_shot, shot_key in enumerate(shots.keys()):
            if max_shots and i_shot==max_shots:
                break
            shot = shots[shot_key]
            shot_intervals = np.array(shot.attrs['shot_intervals'])
            n_elms = shot_intervals.shape[0]
            print(f"Shot {shot_key}  n_ELMs: {n_elms}  min/max ELM times: {shot_intervals.min():.1f} - {shot_intervals.max():.1f} ms")
            # if i_shot == 0:
            #     for key in shot:
            #         data_np = np.array(shot[key])
            #         print(f"  {key}: {data_np.shape} {data_np.dtype}")
            #     for attr in shot.attrs:
            #         print(f"  {attr}: {shot.attrs[attr]}")
            bes_time = np.array(shot['bes_time'])
            bes_signals = np.array(shot['bes_signals'])
            assert bes_signals.shape[1] == bes_time.shape[0]



def plot_elms(
        file: str|Path = '',
        max_elms: int = None,
        merge_pdf: bool = True,
        save: bool = True,
        savedir: str|Path = 'figures',
        downsample: int = 100,
) -> None:

    file = Path(file).absolute()
    print(f"Data file: {file}")
    assert file.exists()

    if save:
        savedir = Path(savedir).absolute()
        os.makedirs(savedir, exist_ok=True)
        os.makedirs(savedir/'pngs', exist_ok=True)

    plt.ioff()
    fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8))
    fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8))
    channel_wise_rms = {i: [] for i in range(64)}
    channel_wise_mean = {i: [] for i in range(64)}

    with h5py.File(file, 'r') as root:
        elms = root['elms']
        n_elms = len(elms)
        print(f"Number of ELMs: {n_elms}")
        for i_elm, elm_key in enumerate(elms.keys()):
            if max_elms and i_elm==max_elms:
                break
            elm = elms[elm_key]
            shot = elm.attrs['shot']
            t_start = elm.attrs['t_start']
            t_stop = elm.attrs['t_stop']
            tag = f"ELM {elm_key} Shot {shot} Time {t_start:.1f}-{t_stop:.1f} ms"
            print(tag)
            bes_time = np.array(elm['bes_time'])
            bes_signals = np.array(elm['bes_signals'])
            assert bes_signals.shape[0] == 64
            assert bes_signals.shape[1] == bes_time.shape[0]
            bes_time_ds = bes_time[::downsample]
            bes_signals_ds = bes_signals[:, ::downsample]
            plt.suptitle(tag)
            for axis1, axis2 in zip(axes1.flat, axes2.flat):
                axis1.clear()
                axis2.clear()
            for chan in range(64):
                chan_d16 = chan // 16  # plot row
                chan_mod8_d2 = (chan % 8) // 2  # plot column
                inter_elm_mask = np.logical_and(bes_time >= t_start, bes_time <= t_stop)
                rms = np.sqrt(np.mean(bes_signals[chan, inter_elm_mask]**2))
                channel_wise_rms[chan].append(rms)
                channel_wise_mean[chan].append(np.mean(bes_signals[chan, inter_elm_mask]))
                plt.sca(axes1[chan_d16, chan_mod8_d2])
                plt.plot(
                    bes_time_ds, 
                    bes_signals_ds[chan, :], 
                    label=f'Ch {chan+1:d} (rms {rms:.2f})',
                    lw=0.75
                )
                plt.sca(axes2[chan_d16, chan_mod8_d2])
                elm_onset_mask = np.abs(bes_time - t_stop) <= 3
                plt.plot(
                    bes_time[elm_onset_mask][::10], 
                    bes_signals[chan, elm_onset_mask][::10], 
                    label=f'Ch {chan+1:d}',
                    lw=0.75
                )
            for axis1, axis2 in zip(axes1.flat, axes2.flat):
                plt.sca(axis1)
                plt.ylim(-10,10)
                plt.axvline(elm.attrs['t_start'], c='k', lw=0.75, ls='--')
                plt.axvline(elm.attrs['t_stop'], c='k', lw=0.75, ls='--')
                plt.legend(fontsize='x-small')
                plt.xlabel("Time (ms)")
                plt.ylabel("Signal (V)")
                plt.sca(axis2)
                plt.ylim(top=10)
                plt.axvline(elm.attrs['t_stop'], c='k', lw=0.75, ls='--')
                plt.legend(fontsize='x-small')
                plt.xlabel("Time (ms)")
                plt.ylabel("Signal (V)")
            for f in [fig1, fig2]:
                plt.figure(f)
                plt.tight_layout()
            if save:
                for f, suffix in zip([fig1, fig2], ['full', 'onset']):
                    f.savefig(savedir/f'elm_{elm_key}_{suffix}.pdf', format='pdf', transparent=True)
                    f.savefig(savedir/f'elm_{elm_key}_{suffix}.png', format='png', dpi=100)
        for f in [fig1, fig2]:
            plt.close(f)

    # merge ELM pdfs
    if save and merge_pdf:
        for suffix in ['full', 'onset']:
            output = savedir / f'all_elms_{suffix}.pdf'
            cmd = f"gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE={output.as_posix()} -dBATCH {savedir.as_posix()}/elm_*_{suffix}.pdf"
            ex = os.system(cmd)
            if ex==0:
                os.system(f"rm -f {savedir.as_posix()}/elm_*_{suffix}.pdf")

    # plot channel-wise mean and RMS
    fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8), sharex=True, sharey=True)
    for chan in channel_wise_mean:
        chan_d16 = chan // 16  # plot row
        chan_mod8_d2 = (chan % 8) // 2  # plot column
        plt.sca(axes2[chan_d16, chan_mod8_d2])
        plt.plot(channel_wise_mean[chan], ls='', marker='.', ms=5, label=f'Ch {chan+1}')
    for axis1 in axes2.flat:
        plt.sca(axis1)
        plt.legend(fontsize='x-small')
    plt.suptitle(f'Channel-wise mean for all ELMs (V)')
    plt.tight_layout()
    if save:
        fig2.savefig(savedir/f'all_mean.pdf', format='pdf', transparent=True)
        fig2.savefig(savedir/f'all_mean.png', format='png', dpi=100)
    plt.close(fig2)

    fig3, axes3 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8), sharex=True, sharey=True)
    for chan in channel_wise_rms:
        chan_d16 = chan // 16  # plot row
        chan_mod8_d2 = (chan % 8) // 2  # plot column
        plt.sca(axes3[chan_d16, chan_mod8_d2])
        plt.semilogy(channel_wise_rms[chan], ls='', marker='.', ms=5, label=f'Ch {chan+1}')
    for axis1 in axes3.flat:
        plt.sca(axis1)
        plt.legend(fontsize='x-small')
    plt.suptitle(f'Channel-wise RMS for all ELMs (V)')
    plt.tight_layout()
    if save:
        fig3.savefig(savedir/f'all_rms.pdf', format='pdf', transparent=True)
        fig3.savefig(savedir/f'all_rms.png', format='png', dpi=100)
    plt.close(fig3)

    if save:
        os.system(f"cd {savedir.as_posix()}; mv -f *.png pngs/")


if __name__=='__main__':
    file = '/home/smithdr/ml/elm_data/step_6_labeled_elm_data/elm_data_v1.hdf5'
    # plot_shots(file, max_shots=10)
    plot_elms(file, max_elms=20)