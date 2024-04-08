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
            max_elms: int = None,
            downsample: int = 100,
            shuffle: bool = False,
            save: bool = False,
            only_onset_discrepency: bool = False,
            estimate_onset: bool = True,
    ):
        fig1, axes1 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8))
        fig2, axes2 = plt.subplots(nrows=4, ncols=4, figsize=(10.5, 8))
        re_response = re.compile(r"\S")

        with h5py.File(self.labeled_elm_data_file, 'r') as root:
            elm_keys = list(root['elms'].keys())
            if shuffle:
                np.random.default_rng().shuffle(elm_keys)
            if max_elms:
                elm_keys = elm_keys[:max_elms]
            for elm_key in elm_keys:
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
                if estimate_onset:
                    # determin ELM onset from last timestamp with signal<=0.2*channel-wise max
                    ch_i_max = np.argmax(onset_signals, axis=1)
                    i_max = int(np.median(ch_i_max))
                    ch_i_onset = np.zeros(ch_i_max.shape, dtype=int)
                    for i_ch in range(onset_signals.shape[0]):
                        pre_onset = np.nonzero(onset_signals[i_ch, 0:i_max]<=0.2*onset_signals[i_ch,i_max])[0]
                        if pre_onset.size:
                            ch_i_onset[i_ch] = pre_onset[-1]
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
                        onset_time-t_stop, 
                        onset_signals[i_ch, :], 
                        label=f'Ch {i_ch+1}',
                        lw=0.75
                    )
                for axis1, axis2 in zip(axes1.flat, axes2.flat):
                    # plot full pre-ELM
                    plt.sca(axis1)
                    plt.ylim(-10,10)
                    plt.axvline(t_start, c='k', lw=0.75, ls='--')
                    plt.axvline(t_stop, c='k', lw=0.75, ls='--')
                    # plot near ELM onset
                    plt.sca(axis2)
                    plt.ylim(top=10)
                    plt.xlim(-1,0.5)
                    plt.axvline(0, c='k', lw=0.75, ls='--')
                    if estimate_onset:
                        plt.axvline(t_onset-t_stop, c='r', lw=0.75, ls='--')
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

    def plot_shot_elm_stats(self, save: bool = False):
        with h5py.File(self.labeled_elm_data_file, 'r') as root:
            elm_keys = list(root['elms'].keys())
            n_elms = len(elm_keys)
            print(f"Number of ELMs: {n_elms}")
            delt = np.empty(n_elms) * np.nan
            elm_ip = np.empty(n_elms) * np.nan
            elm_bt = np.empty(n_elms) * np.nan
            shots = [int(shot) for shot in root['shots']]
            shot_ip = {shot: 0 for shot in shots}
            shot_bt = {shot: 0 for shot in shots}
            shot_nelms = {shot: 0 for shot in shots}
            sum_delt = 0.0
            for i_elm, elm_key in enumerate(elm_keys):
                elm = root['elms'][elm_key]
                shot_num = elm.attrs['shot']
                shot = root['shots'][str(shot_num)]
                assert shot.attrs['ip_pos_phi']==True and shot.attrs['bt_pos_phi']==False, f"ELM {elm_key} has incorrect Ip/Bt direction"
                shot_ip[shot_num] = shot.attrs['ip_extremum']
                shot_bt[shot_num] = shot.attrs['bt_extremum']
                shot_nelms[shot_num] += 1
                elm_ip[i_elm] = shot.attrs['ip_extremum']
                elm_bt[i_elm] = shot.attrs['bt_extremum']
                delt[i_elm] = elm.attrs['t_stop'] - elm.attrs['t_start']
                assert delt[i_elm] > 8.0                
                sum_delt += delt[i_elm]

        for ndarray in [delt, elm_ip, elm_bt]:
            assert np.all(np.isfinite(ndarray))

        print(f"Pre-ELM time (ms):  {self._vector_stats(delt)}")
        print(f"Total pre-ELM time (ms): {int(sum_delt):,d}")
        print(f"Ip (MA):  {self._vector_stats(elm_ip/1e6)}")
        print(f"Bt (T):  {self._vector_stats(elm_bt)}")

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,6))
        plt.suptitle(f"Shot and ELM distributions | {n_elms} ELMs from {len(shots)} shots")

        plt.sca(axes.flat[0])
        plt.hist([np.log10(val) for val in shot_nelms.values()], bins=15)
        plt.xlabel('log10(ELMs per shot)')
        plt.ylabel('Shot count')

        plt.sca(axes.flat[1])
        plt.hist([val/1e6 for val in shot_ip.values()], bins=15)
        plt.xlabel('Ip (MA)')
        plt.ylabel('Shot count')

        plt.sca(axes.flat[2])
        plt.hist([val for val in shot_bt.values()], bins=15)
        plt.xlabel('Bt (T)')
        plt.ylabel('Shot count')

        plt.sca(axes.flat[3])
        plt.hist(np.log10(delt), bins=15)
        plt.xlabel('log10(Pre-ELM time (ms))')
        plt.ylabel('ELM count')

        plt.sca(axes.flat[4])
        plt.hist(elm_ip/1e6, bins=15)
        plt.xlabel('Ip (MA)')
        plt.ylabel('ELM count')

        plt.sca(axes.flat[5])
        plt.hist(elm_bt, bins=15)
        plt.xlabel('Bt (T)')
        plt.ylabel('ELM count')

        for ax in axes.flat:
            plt.sca(ax)
            plt.ylim([1,None])
            plt.yscale('log')

        plt.tight_layout()

        if save:
            fig.savefig(self.save_dir/'shot_elm_stats.pdf', format='pdf', transparent=True)
            fig.savefig(self.save_dir/'pngs'/'shot_elm_stats.png', format='png', dpi=100)

    def plot_channel_stats(self, max_elms: int = None, save: bool = False):
        re_response = re.compile(r"\S")
        with h5py.File(self.labeled_elm_data_file, 'r') as root:
            elms = root['elms']
            elm_keys = list(elms.keys())
            print(f"Number of ELMs: {len(elm_keys)}")
            if max_elms:
                np.random.default_rng().shuffle(elm_keys)
                elm_keys = elm_keys[:max_elms]
            n_elms = len(elm_keys)
            print(f"Number of ELMs for analysis: {n_elms}")
            elm_avg = np.empty((64, n_elms)) * np.nan
            elm_std = np.empty((64, n_elms)) * np.nan
            elm_min = np.empty((64, n_elms)) * np.nan
            elm_max = np.empty((64, n_elms)) * np.nan
            for i_elm, elm_key in enumerate(elm_keys):
                elm = elms[elm_key]
                # shot = root['shots'][str(elm.attrs['shot'])]
                # assert shot.attrs['ip_pos_phi']==True and shot.attrs['bt_pos_phi']==False, f"{i_elm}"
                i_start = np.nonzero(elm['bes_time'] >= elm.attrs['t_start'])[0][0]
                i_stop = np.nonzero(elm['bes_time'] > elm.attrs['t_stop'])[0][0]
                assert i_stop-i_start > 2000
                i_mid = (i_start+i_stop) // 2
                # small window around middle of pre-ELM period
                mid_pre_elm_window = elm['bes_signals'][:, i_mid-1000:i_mid+1000]
                elm_avg[:, i_elm] = np.mean(mid_pre_elm_window, axis=1)
                elm_std[:, i_elm] = np.std(mid_pre_elm_window, axis=1)
                quants = np.quantile(mid_pre_elm_window, [0.005,0.995], axis=1)
                elm_min[:, i_elm] = quants[0, :]
                elm_max[:, i_elm] = quants[1, :]

        for ndarray in [elm_avg, elm_std, elm_min, elm_max]:
            assert np.all(np.isfinite(ndarray))

        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
        elm_indices = [int(elm) for elm in elm_keys]
        for i_ch in range(64):
            fig.suptitle(f"Channel {i_ch+1} stats (middle of pre-ELM phase)")

            for ax in axes.flat:
                plt.sca(ax)
                plt.cla()

            for i, (tag, metric) in enumerate(zip(
                ['Min', 'Avg', 'Max', 'Std'],
                [elm_min, elm_avg, elm_max, elm_std],
            )):
                plt.sca(axes.flat[i])
                if i != 3:
                    plt.hist(metric[i_ch,:], bins=15)
                    plt.xlabel(f'Ch {i_ch+1} {tag} (V)')
                else:
                    plt.hist(np.log10(metric[i_ch,:]), bins=15)
                    plt.xlabel(f'Ch {i_ch+1} log10({tag}) (V)')

                plt.sca(axes.flat[i+4])
                if i != 3:
                    plt.scatter(elm_indices, metric[i_ch,:], s=2**2, marker='.')
                    plt.ylabel(f'Ch {i_ch+1} {tag} (V)')
                else:
                    plt.scatter(elm_indices, np.log10(metric[i_ch,:]), s=2**2, marker='.')
                    plt.ylabel(f'Ch {i_ch+1} log10({tag}) (V)')
                plt.xlabel('ELM index')

            for ax in axes.flat[0:4]:
                plt.sca(ax)
                plt.ylim([1,None])
                plt.yscale('log')
                plt.ylabel('ELM count')

            plt.tight_layout()

            if save:
                fig.savefig(self.save_dir/f'ch_{i_ch+1:02d}_stats.pdf', format='pdf', transparent=True)
                fig.savefig(self.save_dir/'pngs'/f'ch_{i_ch+1:02d}_stats.png', format='png', dpi=100)

            if plt.isinteractive():
                display(fig)
                clear_output(wait=True)
                plt.pause(0.1)
                r = input("<return> to continue or any character to break: ")
                if re_response.match(r):
                    break

        # # merge ELM pdfs
        # if save:
        #     print('Merging PDFs and deleting single PDFs')
        #     cmd = f"gs -q -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE={self.save_dir.as_posix()}/ch_stats.pdf -dBATCH {self.save_dir.as_posix()}/ch_*_stats.pdf"
        #     ex = os.system(cmd)
        #     if ex==0:
        #         os.system(f"rm -f {self.save_dir.as_posix()}/ch_*_stats.pdf")


if __name__=='__main__':
    file = '/home/smithdr/ml/elm_data/step_6_labeled_elm_data/elm_data_v1.hdf5'
    stats = ELM_Data_Stats(file)
    stats.plot_elms(max_elms=10)
    stats.plot_shot_elm_stats()
    stats.plot_channel_stats()