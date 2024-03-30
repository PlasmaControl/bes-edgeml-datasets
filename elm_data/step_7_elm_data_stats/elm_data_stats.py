from pathlib import Path
import os
import dataclasses
import re

import numpy as np
import matplotlib.pyplot as plt
import h5py

from IPython.display import display, clear_output

@dataclasses.dataclass
class ELM_Data_Stats:
    labeled_elm_data_file: str|Path
    save_dir: str|Path = None

    def __post_init__(self):
        self.labeled_elm_data_file = Path(self.labeled_elm_data_file).absolute()
        print(f"Labelled ELM data file: {self.labeled_elm_data_file}")
        assert self.labeled_elm_data_file.exists()

        if not self.save_dir:
            self.save_dir = 'figures'
        self.save_dir = Path(self.save_dir).absolute()
        os.makedirs(self.save_dir/'pngs', exist_ok=True)
        print(f"Save dir for figures: {self.save_dir}")
        assert self.save_dir.exists()

        with h5py.File(self.labeled_elm_data_file, 'r') as root:
            self.shots = np.array([int(shot_key) for shot_key in root['shots']], dtype=int)
            shots2 = np.unique(np.array([int(root['elms'][elm_key].attrs['shot']) for elm_key in root['elms']], dtype=int))
            shot_xor = np.setxor1d(self.shots, shots2, assume_unique=True)
            assert shot_xor.size == 0, "Shots and shots from ELMs are not identical sets"
            print(f"Shots and shots from ELMs are identical sets")
            self.n_shots = len(root['shots'])
            self.n_elms = len(root['elms'])
            n_skipped_shots = np.size(root.attrs['excluded_shots'])

        print(f"Number of shots: {self.n_shots}")
        print(f"Number of ELMs: {self.n_elms}")
        print(f"Number of skipped shots: {n_skipped_shots}")
        print(f"Is interactive?: {plt.isinteractive()}")

    def plot_elms(
            self,
            max_elms: int = 10,
            downsample: int = 100,
            shuffle: bool = False,
            save: bool = False,
            only_onset_discrepency: bool = False,
    ):
        fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8))
        fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8))
        re_response = re.compile(r"\S")

        with h5py.File(self.labeled_elm_data_file, 'r') as root:
            elm_keys = list(root['elms'].keys())
            if shuffle:
                np.random.default_rng().shuffle(elm_keys)
            for i_elm, elm_key in enumerate(elm_keys):
                if max_elms and i_elm==max_elms:
                    break
                elm = root['elms'][elm_key]
                shot = elm.attrs['shot']
                t_start = elm.attrs['t_start']
                t_stop = elm.attrs['t_stop']
                bes_time = np.array(elm['bes_time'])
                bes_signals = np.array(elm['bes_signals'])
                assert bes_signals.shape[0] == 64
                assert bes_signals.shape[1] == bes_time.shape[0]
                onset_mask = np.abs(bes_time - t_stop) <= 3  # time mask near ELM onset
                onset_time = bes_time[onset_mask][::10]
                onset_signals = bes_signals[:, onset_mask][:,::10]
                # determin ELM onset from last timestamp with signal<=0.2*channel-wise max
                ch_i_max = np.argmax(onset_signals, axis=1)
                i_max = int(np.median(ch_i_max))
                ch_i_onset = np.zeros(ch_i_max.shape, dtype=int)
                for i_ch in range(onset_signals.shape[0]):
                    pre_onset = np.nonzero(onset_signals[i_ch, 0:i_max]<=0.2*onset_signals[i_ch,i_max])[0]
                    if pre_onset.size:
                        ch_i_onset[i_ch] = pre_onset[-1]
                    # if np.abs(onset_time[ch_i_onset[i_ch]]-t_stop) > 100:
                    #     ch_i_onset[i_ch] = 0
                i_onset = int(np.median(ch_i_onset))
                t_onset = onset_time[i_onset]
                if only_onset_discrepency and t_stop>=t_onset-0.6 and t_stop<=t_onset+0.3:
                    continue  # skip if onsets are consistent
                for f in [fig1, fig2]:
                    f.suptitle(f"ELM {elm_key} Shot {shot} Time {t_start:.1f}-{t_stop:.1f} ms")
                for ax in list(axes1.flat) + list(axes2.flat):
                    plt.sca(ax)
                    plt.cla()
                for i_ch in range(64):
                    chan_d16 = i_ch // 16  # plot row
                    chan_mod8_d2 = (i_ch % 8) // 2  # plot column
                    # plot full pre-ELM
                    plt.sca(axes1[chan_d16, chan_mod8_d2])
                    plt.plot(
                        bes_time[::downsample], 
                        bes_signals[i_ch, ::downsample], 
                        label=f'Ch {i_ch+1}',
                        lw=0.75
                    )
                    # plot near ELM onset
                    plt.sca(axes2[chan_d16, chan_mod8_d2])
                    plt.plot(
                        onset_time-t_onset, 
                        onset_signals[i_ch, :], 
                        label=f'Ch {i_ch+1}',
                        lw=0.75
                    )
                for axis1, axis2 in zip(axes1.flat, axes2.flat):
                    # plot full pre-ELM
                    plt.sca(axis1)
                    plt.ylim(-10,10)
                    plt.axvline(elm.attrs['t_start'], c='k', lw=0.75, ls='--')
                    plt.axvline(elm.attrs['t_stop'], c='k', lw=0.75, ls='--')
                    # plot near ELM onset
                    plt.sca(axis2)
                    plt.ylim(top=10)
                    # plt.xlim(-1.5,0.5)
                    plt.axvline(0, c='r', lw=0.75, ls='--')
                    plt.axvline(elm.attrs['t_stop']-t_onset, c='k', lw=0.75, ls='--')
                for ax in list(axes1.flat)+list(axes2.flat):
                    plt.sca(ax)
                    plt.legend(fontsize='x-small')
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Signal (V)")
                for f in [fig1, fig2]:
                    f.tight_layout()
                if save:
                    for f, suffix in zip([fig1, fig2], ['full', 'onset']):
                        f.savefig(self.save_dir/f'elm_{elm_key}_{suffix}.pdf', format='pdf', transparent=True)
                        f.savefig(self.save_dir/'pngs'/f'elm_{elm_key}_{suffix}.png', format='png', dpi=100)
                if plt.isinteractive():
                    for f in [fig1, fig2]:
                        display(f)
                    clear_output(wait=True)
                    plt.pause(0.5)
                    r = input("<return> to continue or any character to break: ")
                    if re_response.match(r):
                        break

        # merge ELM pdfs
        if save:
            print('Merging PDFs and deleting single PDFs')
            for suffix in ['full', 'onset']:
                output = self.save_dir / f'all_elms_{suffix}.pdf'
                cmd = f"gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE={output.as_posix()} -dBATCH {self.save_dir.as_posix()}/elm_*_{suffix}.pdf"
                ex = os.system(cmd)
                if ex==0:
                    os.system(f"rm -f {self.save_dir.as_posix()}/elm_*_{suffix}.pdf")

    @staticmethod
    def _vector_stats(vector:np.ndarray):
        quants = np.quantile(vector, [0.05,0.95])
        output =  f"min {vector.min():<5.2f} "
        output += f"q5 {quants[0]:<5.2f} "
        output += f"med {np.median(vector):<5.2f} "
        output += f"q95 {quants[1]:<5.2f} "
        output += f"max {np.max(vector):<5.2f} "
        return output

    def plot_shot_elm_stats(
            self, 
            max_elms: int = 200,
            save: bool = False,
    ):
        with h5py.File(self.labeled_elm_data_file, 'r') as root:
            elms = root['elms']
            n_elms = len(elms)
            shots = [int(shot) for shot in root['shots']]
            print(f"Number of ELMs: {n_elms}")
            n_elms = n_elms if max_elms is None else max_elms
            print(f"Number of ELMs for analysis: {n_elms}")
            delt = np.empty(n_elms) * np.nan
            elm_ip = np.empty(n_elms) * np.nan
            elm_bt = np.empty(n_elms) * np.nan
            shot_ip = {shot: 0 for shot in shots}
            shot_bt = {shot: 0 for shot in shots}
            shot_nelms = {shot: 0 for shot in shots}
            # elm_avg = np.empty((64, n_elms)) * np.nan
            # elm_std = np.empty((64, n_elms)) * np.nan
            # elm_min = np.empty((64, n_elms)) * np.nan
            # elm_max = np.empty((64, n_elms)) * np.nan
            elm_keys = list(elms.keys())
            rng = np.random.default_rng()
            if max_elms:
                rng.shuffle(elm_keys)
            for i_elm, elm_key in enumerate(elm_keys):
                if max_elms and i_elm==max_elms:
                    break
                elm = elms[elm_key]
                shot = root['shots'][str(elm.attrs['shot'])]
                assert shot.attrs['ip_pos_phi']==True and shot.attrs['bt_pos_phi']==False, f"{i_elm}"
                shot_ip[elm.attrs['shot']] = shot.attrs['ip_extremum']
                shot_bt[elm.attrs['shot']] = shot.attrs['bt_extremum']
                shot_nelms[elm.attrs['shot']] += 1
                elm_ip[i_elm] = shot.attrs['ip_extremum']
                elm_bt[i_elm] = shot.attrs['bt_extremum']
                delt[i_elm] = elm.attrs['t_stop'] - elm.attrs['t_start']
                assert delt[i_elm] > 8.0
                # i_start = np.nonzero(elm['bes_time'] >= elm.attrs['t_start'])[0][0]
                # i_stop = np.nonzero(elm['bes_time'] > elm.attrs['t_stop'])[0][0]
                # assert i_stop-i_start > 8000
                # t_mid = elm.attrs['t_start'] + delt[i_elm]/2
                # i_mid = (i_start+i_stop) // 2
                # channel_windows = elm['bes_signals'][:, i_mid-500:i_mid+500]
                # elm_avg[:, i_elm] = 0  # np.mean(channel_windows, axis=1)
                # elm_std[:, i_elm] = 0  # np.std(channel_windows, axis=1)
                # elm_min[:, i_elm] = 0  # np.min(channel_windows, axis=1)
                # elm_max[:, i_elm] = 0  # np.max(channel_windows, axis=1)
                # if i_elm==0:
                #     print(elm.attrs.keys())
                #     print(elm.keys())
                #     print(shot.attrs.keys())
                #     print(shot.attrs['ip_pos_phi'], shot.attrs['bt_pos_phi'])

        for ndarray in [delt, elm_ip, elm_bt]:
            assert np.all(np.isfinite(ndarray))

        print(f"delt (ms):  {self._vector_stats(delt)}")
        print(f"Ip (MA):  {self._vector_stats(elm_ip/1e6)}")
        print(f"Bt (T):  {self._vector_stats(elm_bt)}")

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,6))

        plt.sca(axes.flat[0])
        plt.hist([np.log10(val) for val in shot_nelms.values()], bins=15)
        plt.xlabel('log10(ELMs per shot)')
        plt.ylabel('Shot count')
        plt.ylim([1,None])
        plt.yscale('log')

        plt.sca(axes.flat[1])
        plt.hist([val/1e6 for val in shot_ip.values()], bins=15)
        plt.xlabel('Ip (MA)')
        plt.ylabel('Shot count')
        plt.ylim([1,None])
        plt.yscale('log')

        plt.sca(axes.flat[2])
        plt.hist([val for val in shot_bt.values()], bins=15)
        plt.xlabel('Bt (T)')
        plt.ylabel('Shot count')
        plt.ylim([1,None])
        plt.yscale('log')

        plt.sca(axes.flat[3])
        plt.hist(np.log10(delt), bins=15)
        plt.xlabel('log10(Pre-ELM time (ms))')
        plt.ylabel('ELM count')
        plt.ylim([1,None])
        plt.yscale('log')

        plt.sca(axes.flat[4])
        plt.hist(elm_ip/1e6, bins=15)
        plt.xlabel('Ip (MA)')
        plt.ylabel('ELM count')
        plt.ylim([1,None])
        plt.yscale('log')

        plt.sca(axes.flat[5])
        plt.hist(elm_bt, bins=15)
        plt.xlabel('Bt (T)')
        plt.ylabel('ELM count')
        plt.ylim([1,None])
        plt.yscale('log')

        plt.tight_layout()

    def plot_channel_stats(self, save=False):
        pass


if __name__=='__main__':
    file = '/home/smithdr/ml/elm_data/step_6_labeled_elm_data/elm_data_v1.hdf5'
    stats = ELM_Data_Stats(file)
    # stats.plot_elms(save=False, only_onset_discrepency=True, max_elms=10)
    stats.plot_shot_elm_stats()