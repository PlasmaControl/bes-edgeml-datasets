from pathlib import Path
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.signal, scipy.fft
import h5py
import ipywidgets as widgets

@dataclasses.dataclass(eq=False)
class BES_ELM_Labeling_App:
    data_hdf5_file: str|Path
    elm_labels_hdf5_file: str|Path
    truncate: bool = False

    def __post_init__(self):

        self.data_hdf5_file = Path(self.data_hdf5_file)
        self.elm_labels_hdf5_file = Path(self.elm_labels_hdf5_file).absolute()

        # prepare MPL figure
        with plt.ioff():
            self.fig, self.axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(8.5,6.5))
        self.axes[-1].set_xlabel('Time (ms)')
        self.axes[-1].set_ylabel('Frequency (kHz)')
        self.fig.suptitle('Shot')
        self.canvas = self.fig.canvas
        self.canvas.header_visible = False
        self.canvas.footer_visible = False
        self.canvas.toolbar_visible = True
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click_callback)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move_callback)
        self.toolbar = self.canvas.toolbar

        with h5py.File(self.data_hdf5_file) as root:
            shots = [int(group_name) for group_name in root['shots']]
        print(f"HDF5 data file: {self.data_hdf5_file}")
        print(f"  Shots: {len(shots)}")
        numpy.random.default_rng().shuffle(shots)
        # for shot in [179453, 189113, 191672, 179873, 166434]:
        #     shots.insert(0, shot)

        # create if needed, and truncate or append
        if self.truncate:
            print('Truncating labeled ELM file')
        with h5py.File(self.elm_labels_hdf5_file, 'w' if self.truncate else 'a') as root:
            if 'shots' not in root:
                root.create_group(name='shots')
            if 'elms' not in root:
                root.create_group(name='elms')
            if 'excluded_shots' not in root.attrs:
                root.attrs['excluded_shots'] = np.array([], dtype=int)
            assert 'shots' in root
            assert 'elms' in root
            assert 'excluded_shots' in root.attrs
            print(f"HDF5 labeled ELM file: {self.elm_labels_hdf5_file}")
            print(f"  {len(root['elms'])} ELMs from {len(root['shots'])} shots ({len(root.attrs['excluded_shots'])} excluded shots)")
            for shot in root['shots']:
                shots.remove(int(shot))
            for shot in root.attrs['excluded_shots']:
                shots.remove(shot)

        self.shot = None
        self.next_shot = (shot for shot in shots)
        self.mouse_guide_lines = []
        self.start_click_tmp_line = []
        self.t_start = None
        self.t_stop = None
        self.vspans = [ [] for ax in self.axes ]
        self.peak_times = []
        self.elm_cycle_spans = []  # list of MPL spans for ELM cycles
        self.elm_cycle_intervals = []  # list of t_start, t_stop pairs for ELM cycles

        # prepare ipywidgets
        def_layout = {'width':'80%', 'margin':'5px'}
        load_new_shot_button = widgets.Button(description='Save & load new', layout=def_layout)
        load_new_shot_button.on_click(self.load_new_shot)
        reset_button = widgets.Button(description='Reset view', layout=def_layout)
        reset_button.on_click(self.toolbar.home)
        back_button = widgets.Button(description='Previous view', layout=def_layout)
        back_button.on_click(self.toolbar.back)
        self.pan_button = widgets.Button(description='Pan view', layout=def_layout)
        self.pan_button.on_click(self.pan_callback)
        pan_left_button = widgets.Button(description='Pan left', layout=def_layout)
        pan_left_button.on_click(self.pan_left_callback)
        pan_right_button = widgets.Button(description='Pan right', layout=def_layout)
        pan_right_button.on_click(self.pan_right_callback)
        self.zoom_button = widgets.Button(description='Zoom selection', layout=def_layout)
        self.zoom_button.on_click(self.zoom_callback)
        zoom_in_button = widgets.Button(description='Zoom in', layout=def_layout)
        zoom_in_button.on_click(self.zoom_in_callback)
        zoom_out_button = widgets.Button(description='Zoom out', layout=def_layout)
        zoom_out_button.on_click(self.zoom_out_callback)
        autoy_button = widgets.Button(description='Autoscale y', layout=def_layout)
        autoy_button.on_click(self.autoy_callback)
        self.is_delete_active = False
        self.delete_button = widgets.Button(
            description='Delete', 
            layout={'width':'80%', 'margin':'5px 5px 20px 5px'},
        )
        self.delete_button.on_click(self.delete_callback)

        self.mode_selection = widgets.RadioButtons(
            options=['Auto', 'Manual'],
            description='Selection mode:',
            layout={'width': '80%'}
        )
        
        self.status_label = widgets.Label(value='State: GUI launched')
        self.controls = widgets.VBox(
            layout = {'justify_content':'center', 'align_items':'center'},
            children = [
                load_new_shot_button,
                reset_button,
                back_button,
                self.pan_button,
                pan_left_button,
                pan_right_button,
                self.zoom_button,
                zoom_in_button,
                zoom_out_button,
                autoy_button,
                self.delete_button,
                self.mode_selection,
            ]
        )

        # load first shot
        self.load_new_shot()
        self.fig.tight_layout(h_pad=0.6)

    def delete_callback(self, *_):
        if self.toolbar.mode.startswith(('zoom','pan')):
            self.is_delete_active = False
        else:
            self.is_delete_active = not self.is_delete_active
        self.status()

    def pan_callback(self, *_):
        self.toolbar.pan()
        self.is_delete_active = False
        self.status()

    def zoom_callback(self, *_):
        self.toolbar.zoom()
        self.is_delete_active = False
        self.status()

    def zoom_in_callback(self, *_):
        xlim = self.axes[-1].get_xlim()
        x_middle, x_range = np.mean(xlim), xlim[1]-xlim[0]
        x_range /= 2
        self.axes[-1].set_xlim(np.array([-1,1])*x_range/2 + x_middle)

    def zoom_out_callback(self, *_):
        xlim = self.axes[-1].get_xlim()
        x_middle, x_range = np.mean(xlim), xlim[1]-xlim[0]
        x_range *= 2
        self.axes[-1].set_xlim(np.array([-1,1])*x_range/2 + x_middle)

    def pan_right_callback(self, *_):
        xlim = self.axes[-1].get_xlim()
        x_middle, x_range = np.mean(xlim), xlim[1]-xlim[0]
        x_middle += x_range/2
        self.axes[-1].set_xlim(np.array([-1,1])*x_range/2 + x_middle)

    def pan_left_callback(self, *_):
        xlim = self.axes[-1].get_xlim()
        x_middle, x_range = np.mean(xlim), xlim[1]-xlim[0]
        x_middle -= x_range/2
        self.axes[-1].set_xlim(np.array([-1,1])*x_range/2 + x_middle)

    def autoy_callback(self, b):
        for ax in self.axes[:-1]:
            ax.relim(visible_only=True)
            ax.autoscale(axis='y', enable=True)
        self.axes[-1].set_ylim(0,100)

    def on_mouse_move_callback(self, mouse_event):
        if self.toolbar.mode.startswith(('pan','zoom')):
            return
        for line in self.mouse_guide_lines:
            line.set_xdata([mouse_event.xdata, mouse_event.xdata])

    def status(self):
        self.zoom_button.button_style = 'info' if self.toolbar.mode.startswith('zoom') else ''
        self.pan_button.button_style = 'info' if self.toolbar.mode.startswith('pan') else ''
        self.delete_button.button_style = 'warning' if self.is_delete_active else ''
        if self.toolbar.mode.startswith('pan'):
            self.status_label.value = 'State: pan mode'
        elif self.toolbar.mode.startswith('zoom'):
            self.status_label.value = 'State: zoom mode'
        elif self.is_delete_active:
            self.status_label.value = 'State: delete mode'
        elif self.t_start is None:
            self.status_label.value = 'State: click ELM cycle start'
        else:
            self.status_label.value = 'State: click ELM cycle end'

    def save_elm_cycles_for_shot(self):
        if not self.shot or not self.elm_cycle_intervals:
            return
        assert len(self.elm_cycle_intervals) == len(self.elm_cycle_spans)
        with (
            h5py.File(self.elm_labels_hdf5_file, 'a') as labels_file,
            h5py.File(self.data_hdf5_file, 'r') as data_file,
        ):
            shot_str = str(self.shot)
            shot_data_group = data_file['shots'][shot_str]
            pinj_time = np.array(shot_data_group['pinj_time'])
            pinj_15l = np.array(shot_data_group['pinj_15l'])/1e6
            for i_interval, interval in reversed(list(enumerate(self.elm_cycle_intervals))):
                t_start, t_stop = interval
                time_mask = np.logical_and(
                    pinj_time >= t_start,
                    pinj_time <= t_start,
                )
                if np.any(pinj_15l[time_mask] <= 0.5):
                    self.elm_cycle_intervals.pop(i_interval)
                    print(f"Skipping interval {t_start:.1-}")
            if not self.elm_cycle_intervals:
                return
            assert shot_str not in labels_file['shots']
            shot_group = labels_file['shots'].create_group(name=shot_str)
            for key, value in shot_data_group.attrs.items():
                shot_group.attrs[key] = value
            # for key, value in shot_data_group.items():
            #     shot_group.create_dataset(name=key, data=value)
            assert 'shot_intervals' not in shot_group
            shot_intervals = []
            elms = [int(elm_str) for elm_str in labels_file['elms']]
            next_i_elm = np.max(elms) + 1 if elms else 0
            for t_start, t_stop in self.elm_cycle_intervals:
                t_mid = (t_start+t_stop)/2
                skip = False
                for interval in shot_intervals:
                    if t_mid >=interval[0] and t_mid <= interval[1]:
                        skip = True
                        break
                if skip:
                    continue
                elm_group = labels_file['elms'].create_group(f"{next_i_elm:06d}")
                elm_group.attrs['shot'] = self.shot
                elm_group.attrs['t_start'] = t_start
                elm_group.attrs['t_stop'] = t_stop
                shot_intervals.append([t_start, t_stop])
                next_i_elm += 1
            shot_group.attrs['shot_intervals'] = np.array(shot_intervals)

    def load_new_shot(self, *_):
        self.status_label.value = 'State: saving and loading new shot...'
        if self.shot and self.elm_cycle_intervals:
            self.save_elm_cycles_for_shot()
        else:
            if self.shot:
                with h5py.File(self.elm_labels_hdf5_file, 'a') as root:
                    root.attrs['excluded_shots'] = np.append(root.attrs['excluded_shots'], self.shot)
        # load new shot
        with h5py.File(self.elm_labels_hdf5_file, 'a') as root:
            shots = [int(shot_str) for shot_str in root['shots']]
            shots.extend(root.attrs['excluded_shots'].tolist())
        while True:
            self.shot = next(self.next_shot)
            if self.shot not in shots:
                break
        self.fig.suptitle(f'Shot {self.shot}')
        self.elm_cycle_spans = []
        self.elm_cycle_intervals = []
        with h5py.File(self.data_hdf5_file) as data_root:
            group = data_root['shots'][str(self.shot)]
            self.mouse_guide_lines = []
            self.vspans = [ [] for ax in self.axes ]
            for ax in self.axes:
                ax.clear()
            # Ip and Pinj
            ip_time = np.array(group['ip_time'])
            ip = np.array(group['ip'])/1e6
            self.axes[0].plot(ip_time, ip, label='Ip (MA)')
            self.axes[0].plot(np.array(group['pinj_time']), np.array(group['pinj'])/1e3/10, label='PINJ/10 (MW)')
            self.axes[0].plot(np.array(group['pinj_time']), np.array(group['pinj_15r'])/1e6, label='PINJ_15R (MW)')
            self.axes[0].plot(np.array(group['pinj_time']), np.array(group['pinj_15l'])/1e6, label='PINJ_15L (MW)')
            self.axes[0].set_ylabel('Ip and Pnbi')
            # interferometer line-averaged density
            denv_time = np.array(group['denv3f_time'])
            denv_names = ['denv3f']
            denv_signals = np.array([group[denv_channel] for denv_channel in denv_names])
            self.axes[1].plot(denv_time, denv_signals.T, label=denv_names[0])
            self.axes[1].set_ylabel('Line avg ne')
            # D_alpha filterscopes
            fs_time = np.array(group['FS_time'])
            fs_names = ['FS03','FS04','FS05']
            fs_signals = np.array([group[fs_channel] for fs_channel in fs_names])
            self.axes[2].plot(fs_time, fs_signals.T, label=fs_names,)
            self.axes[2].set_ylabel('Da (au)')
            # auto ELM labeler from filterscope
            fs_channel_mean = np.mean(fs_signals, axis=1)
            max_fs_channel = np.argmax(fs_channel_mean)
            auto_elm_signal = fs_signals[max_fs_channel, :]
            boxcar_window = scipy.signal.get_window('boxcar', 5)
            auto_elm_signal = scipy.signal.convolve(auto_elm_signal, boxcar_window, mode='same')
            d_signal = np.diff(auto_elm_signal)
            self.axes[2].plot(fs_time[1:], d_signal, lw=0.5, c='k', label='diff')
            i_peaks, peak_properties = scipy.signal.find_peaks(
                d_signal,
                distance=int(20/(fs_time[1]-fs_time[0])),
                height=1e-3,
                prominence=1e-3,
            )
            max_height = peak_properties['peak_heights'].max()
            max_prom = peak_properties['prominences'].max()
            tmp = np.logical_and(
                peak_properties['prominences'] >= (max_prom/25),
                peak_properties['peak_heights'] >= (max_height/25),
            )
            self.peak_times = fs_time[i_peaks[tmp]]
            for i in [1,2,3]:
                for peak_time in self.peak_times:
                    self.axes[i].axvline(x=peak_time, c='k', ls='--', lw=0.5)
            # BES signals
            bes_time = np.array(group['bes_time'])
            bes_mean_signal = np.mean(np.array(group['bes_signals']), axis=0, keepdims=False)
            self.axes[3].plot(
                bes_time, 
                bes_mean_signal,
                label=['BES mean'],
            )
            self.axes[3].set_ylabel('BES (V)')
            # BES spectrogram
            with scipy.fft.set_workers(4):
                f, _, Sxx = scipy.signal.spectrogram(  # f in kHz
                    x=bes_mean_signal,
                    fs=(bes_time.size-1) / (bes_time[-1]-bes_time[0]),  # kHz
                    window='hann',
                    nperseg=128,
                    noverlap=128//2,
                )
            Sxx = np.log10(Sxx+1e-9)
            self.axes[4].imshow(
                Sxx, 
                vmax=Sxx.max()-3,
                vmin=Sxx.max()-6,
                aspect='auto',
                origin='lower',
                extent=[bes_time[0], bes_time[-1], f[0], f[-1]],
            )
            self.axes[4].set_ylim(0, 100)
            self.axes[4].set_ylabel('Frequency (kHz)')
            self.axes[4].set_xlabel('Time (ms)')
            ip_end_index = np.flatnonzero(ip > 0.1)[-1]
            t_end = ip_time[ip_end_index] + 500
            self.axes[0].set_xlim([0, t_end])
            # finalize plot
            self.mouse_guide_lines = [ax.axvline(x=np.nan, ls='--', c='m') for ax in self.axes]
            self.start_click_tmp_line = [ax.axvline(x=np.nan, c='m') for ax in self.axes]
            for ax in self.axes[:-1]:
                ax.legend(fontsize='small', loc='upper right', labelspacing=0.2)
                ax.relim()
            self.status()

    def append_elm_cycle(self, t_start, t_stop):
        assert len(self.elm_cycle_intervals) == len(self.elm_cycle_spans)
        if t_stop <= t_start:
            return
        t_mid = (t_start + t_stop)/2
        for t_start_stop in self.elm_cycle_intervals:
            if t_mid >= t_start_stop[0] and t_mid <= t_start_stop[1]:
                return
        self.elm_cycle_intervals.append([t_start, t_stop])
        self.elm_cycle_spans.append(
            [ax.axvspan(t_start, t_stop, alpha=0.1, color='m') for ax in self.axes]
        )
        assert len(self.elm_cycle_intervals) == len(self.elm_cycle_spans)

    def delete_elm_cycle(self, t_delete: float):
        assert len(self.elm_cycle_intervals) == len(self.elm_cycle_spans)
        self.t_start = None
        self.t_stop = None
        for line in self.start_click_tmp_line:
            line.set_xdata([np.nan, np.nan])
        n_elm_cycles = len(self.elm_cycle_intervals)
        for i_elm in range(n_elm_cycles):
            t_start, t_stop = self.elm_cycle_intervals[i_elm]
            if t_delete >= t_start and t_delete <= t_stop:
                self.elm_cycle_intervals.pop(i_elm)
                for span in self.elm_cycle_spans[i_elm]:
                    span.remove()
                self.elm_cycle_spans.pop(i_elm)
                break
        assert len(self.elm_cycle_intervals) == len(self.elm_cycle_spans)
        self.status()

    def on_mouse_click_callback(self, mouse_event):
        if self.toolbar.mode.startswith(('pan','zoom')) or len(self.mouse_guide_lines)==0:
            return
        if self.is_delete_active:
            self.delete_elm_cycle(mouse_event.xdata)
            return
        # otherwise, mouse clicks are marking ELMs (auto or manual)
        if self.t_start is None:
            self.t_start = mouse_event.xdata
            for line in self.start_click_tmp_line:
                line.set_xdata([self.t_start, self.t_start])
        else:
            self.t_stop = mouse_event.xdata
            for line in self.start_click_tmp_line:
                line.set_xdata([np.nan, np.nan])
        if not self.t_start or not self.t_stop:
            return
        if self.mode_selection.index == 0:
            # auto ELM (1 or more) start/stop segmentation
            assert self.t_start and self.t_stop
            tmp = np.logical_and(
                self.peak_times >= self.t_start,
                self.peak_times <= self.t_stop,
            )
            window_peak_times = self.peak_times[tmp]
            for i in range(window_peak_times.size):
                if i == 0: continue
                t_start = window_peak_times[i-1] + 1  # ms after previous ELM onset
                t_stop = window_peak_times[i] - 0.1  # ms before ELM onset
                self.append_elm_cycle(t_start, t_stop)
        else:
            # manual single ELM start/stop
            t_mean = np.mean([self.t_start, self.t_stop])
            for t_start_stop in self.elm_cycle_intervals:
                if t_mean > t_start_stop[0] and t_mean < t_start_stop[1]:
                    # ELM cycle already exists
                    return
            self.append_elm_cycle(self.t_start, self.t_stop)
        self.t_start = None
        self.t_stop = None
        self.status()
