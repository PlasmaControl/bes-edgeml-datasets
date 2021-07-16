import time as timelib
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
import MDSplus
import h5py
from edgeml import bes2hdf5

os.chdir('/fusion/projects/diagnostics/bes/smithdr/labeled-elms')

repo_directory = bes2hdf5.repo_directory
elm_signals_file = repo_directory / 'edgeml/elm/workflow/data/unlabeled-elm-events/elm-events.hdf5'

elm_labeling_directory = Path().absolute()
valid_users = ['smithdr', 'mckee', 'yanz', 'burkem']
user_name = os.environ['USER']
user_index = valid_users.index(user_name)
user_data_directory = Path(elm_labeling_directory / 'data' / user_name)
user_data_directory.mkdir(parents=True, exist_ok=True)

labeled_elms_file = user_data_directory / f'labeled-elm-events-{user_name}.hdf5'


class ElmTaggerGUI(object):

    def __init__(self, save_pdf=False):
        self.time_markers = [None, None, None, None]
        self.marker_label = ['Pre-ELM start',
                             'ELM start',
                             'ELM end',
                             'Post-ELM end']
        self.fig, self.axes = plt.subplots(ncols=3, figsize=[11.5, 4])
        self.fig_num = self.fig.number
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        for axes in self.axes:
            axes.set_xlabel('Time (ms)')
            axes.set_title('empty')
        self.axes[0].set_ylabel('Line avg dens (AU)')
        self.axes[1].set_ylabel('D alpha')
        self.axes[2].set_ylabel('BES')
        plt.tight_layout(rect=(0, 0.15, 1, 1))

        self.fig.canvas.mpl_connect('button_press_event', self.axis_click)

        self.save_pdf = save_pdf

        # self.quit_button = widgets.Button(plt.axes([0.01, 0.05, 0.07, 0.075]),
        #                                   'Quit',
        #                                   color='lightsalmon',
        #                                   hovercolor='red')
        # self.quit_button.on_clicked(self.on_close)

        self.skip_button = widgets.Button(plt.axes([0.01, 0.05, 0.07, 0.075]),
                                          'Skip ELM',
                                          color='lightsalmon',
                                          hovercolor='red')
        self.skip_button.on_clicked(self.skip)

        self.clear_button = widgets.Button(plt.axes([0.2, 0.05, 0.2, 0.075]),
                                           'Clear markers',
                                           color='lightgray',
                                           hovercolor='darkgray')
        self.clear_button.on_clicked(self.clear_markers)

        self.status_box = widgets.TextBox(plt.axes([0.5, 0.05, 0.2, 0.075]),
                                         'Status:',
                                          color='w',
                                          hovercolor='w')

        self.accept_button = widgets.Button(plt.axes([0.75, 0.05, 0.2, 0.075]),
                                            'Accept markers',
                                            color='whitesmoke',
                                            hovercolor='whitesmoke')
        self.accept_button.on_clicked(self.accept)

        self.multi = widgets.MultiCursor(self.fig.canvas, self.axes, color='r', lw=1)

        self.elm_data_file = h5py.File(elm_signals_file, 'r')
        self.nelms = len(self.elm_data_file)

        self.label_file = h5py.File(labeled_elms_file, 'a')
        self.labeled_elms = self.label_file.attrs.setdefault('labeled_elms',
                                                             np.array([], dtype=np.int))
        self.skipped_elms = self.label_file.attrs.setdefault('skipped_elms',
                                                             np.array([], dtype=np.int))

        self.validate_data_file()

        self.rng = np.random.default_rng()
        self.elm_index = None
        self.shot = None
        self.start_time = None
        self.stop_time = None
        self.time = None
        self.signals = None
        self.connection = MDSplus.Connection('atlas.gat.com')

        self.vlines = []
        self.data_lines = []
        self.clear_and_get_new_elm()

    def validate_data_file(self):
        if len(self.label_file) != self.labeled_elms.size:
            self.labeled_elms = np.array([int(elm_index) for elm_index in self.label_file],
                                         dtype=np.int)
        for elm_index in self.label_file:
            assert(int(elm_index) in self.labeled_elms)
            assert(int(elm_index) not in self.skipped_elms)
        for elm_index in np.concatenate((self.labeled_elms, self.skipped_elms)):
            assert ((elm_index - user_index) % 4 == 0)
        print('Labeled data file is valid')

    def on_close(self, *args, **kwargs):
        print('on_close')
        self.validate_data_file()
        if self.label_file:
            self.label_file.close()
        if self.elm_data_file:
            self.elm_data_file.close()
        # if self.fig and plt.fignum_exists(self.fig.number):
        #     plt.close(self.fig)
        bes2hdf5.traverse_h5py(labeled_elms_file)

    # def __del__(self):
    #     self.fig.close()

    def skip(self, event):
        # log ELM index, then clear and get new ELM
        self.skipped_elms = np.append(self.skipped_elms, self.elm_index)
        self.label_file.attrs['skipped_elms'] = self.skipped_elms
        self.clear_and_get_new_elm()
        plt.draw()

    def clear_markers(self, event):
        # clear marker data, reset appearance
        self.remove_vlines()
        self.time_markers = [None, None, None, None]
        self.set_status()
        self.accept_button.color = 'whitesmoke'
        self.accept_button.hovercolor = 'whitesmoke'
        plt.draw()

    def accept(self, event):
        if self.save_pdf:
            pdf_file = user_data_directory / \
                       f'elm_{self.elm_index:05d}_shot_{self.shot}.pdf'
            plt.savefig(pdf_file, format='pdf', transparent=True)
        self.labeled_elms = np.append(self.labeled_elms, self.elm_index)
        self.label_file.attrs['labeled_elms'] = self.labeled_elms
        print(f'Labeled ELMs: {self.labeled_elms.size}')
        self.log_elm_markers()
        self.clear_and_get_new_elm()
        plt.draw()

    def log_elm_markers(self):
        time = self.time
        signals = self.signals
        mask = np.logical_and(time >= self.time_markers[0],
                              time <= self.time_markers[3])
        time = time[mask]
        signals = signals[:,mask]
        labels = np.zeros(time.shape, dtype=np.int8)
        mask = np.logical_and(time >= self.time_markers[1],
                              time <= self.time_markers[2])
        labels[mask] = 1

        groupname = f'{self.elm_index:05d}'
        assert(groupname not in self.label_file)
        elm_group = self.label_file.create_group(groupname)
        elm_group.attrs['shot'] = self.shot
        elm_group.create_dataset('time', data=time)
        elm_group.create_dataset('signals', data=signals)
        elm_group.create_dataset('labels', data=labels)


    def clear_and_get_new_elm(self, event=None):
        # plots: remove vlines, remove data lines and legend
        self.remove_vlines()
        self.remove_data_lines_and_legend()
        # markers: reset time markers, reset status, reset accept button
        self.time_markers = [None, None, None, None]
        self.set_status()
        self.accept_button.color = 'whitesmoke'
        self.accept_button.hovercolor = 'whitesmoke'
        # data: get new ELM instance, plot new data
        rng_range = self.nelms//4-1
        while True:
            candidate_index = self.rng.integers(0, rng_range)*4 + user_index
            if (candidate_index in self.labeled_elms) or \
                    (candidate_index in self.skipped_elms):
                continue
            if f'{candidate_index:05d}' not in self.elm_data_file:
                continue
            self.elm_index = candidate_index
            break
        assert((self.elm_index-user_index)%4 == 0)
        elm_group = self.elm_data_file[f'{self.elm_index:05d}']
        self.shot = elm_group.attrs['shot']
        self.time = elm_group['time'][:]
        self.signals = elm_group['signals'][:,:]
        self.start_time = self.time[0]
        self.stop_time = self.time[-1]
        print(f'ELM index {self.elm_index}  shot {self.shot}')
        for axes in self.axes:
            axes.set_prop_cycle(None)
        self.plot_density()
        self.plot_bes()
        self.plot_dalpha()
        for axes in self.axes:
            plt.sca(axes)
            plt.legend(loc='upper right')
            plt.title(f'Shot {self.shot} | ELM index {self.elm_index}')
            plt.xlim(self.start_time, self.stop_time)
        plt.draw()

    def plot_density(self):
        node_specs = [
                        # [r'\denr0f', ''],
                        [r'\denv2f', ''],
                        [r'\denv3f', ''],
                        [r'\ripzb1phi', r'\rpich2phi'],
                        ]
        self.connection.openTree('electrons', self.shot)
        plt.sca(self.axes[0])
        for node_spec in node_specs:
            for data_node in node_spec:
                if not data_node:
                    continue
                data = None
                time = None
                try:
                    t1 = timelib.time()
                    data = np.array(self.connection.get(data_node))
                    time = np.array(self.connection.get(f'dim_of({data_node})'))
                    t2 = timelib.time()
                    print(f'  Elapsed time for {data_node}: {t2 - t1:.1f} s')
                except:
                    pass
                if data is None or time is None:
                    continue
                mask = np.logical_and(time >= self.start_time,
                                      time <= self.stop_time)
                if np.count_nonzero(mask)<10:
                    continue
                data = data[mask]
                time = time[mask]
                decimate = data.size // 1500 + 1
                data = data[::decimate]
                time = time[::decimate]
                line = plt.plot(time, data/data.max(), label=data_node)
                self.data_lines.append(line)
                break
        plt.ylim(0.7,1.1)
        self.connection.closeTree('electrons', self.shot)

    def plot_bes(self):
        plt.sca(self.axes[2])
        decimate = self.time.size // 1500 + 1
        time = self.time[::decimate]
        for i_signal in [20, 22, 24]:
            data = self.signals[i_signal-1, ::decimate]
            line = plt.plot(time, data, label=f'Ch {i_signal}')
            self.data_lines.append(line)

    def plot_dalpha(self):
        ptnames = ['FS02', 'FS03', 'FS04',
                   # 'FS05', 'FS06', 'FS07', 'FS08',
                   # 'FS02UP', 'FS03UP', 'FS04UP',
                   ]
        plt.sca(self.axes[1])
        self.connection.openTree('spectroscopy', self.shot)
        for ptname in ptnames:
            data = None
            time = None
            data_tag = f'\\{ptname}'
            try:
                t1 = timelib.time()
                data = np.array(self.connection.get(data_tag))
                time = np.array(self.connection.get(f'dim_of({data_tag})'))
                t2 = timelib.time()
                print(f'  Elapsed time for {data_tag}: {t2 - t1:.1f} s')
            except MDSplus._mdsshr.MdsException:
                print(f'  FAILED: {data_tag}')
            if data is None or time is None:
                continue
            mask = np.logical_and(time >= self.start_time,
                                  time <= self.stop_time)
            data = data[mask]
            time = time[mask]
            decimate = data.size // 1500 + 1
            data = data[::decimate]
            time = time[::decimate]
            line = plt.plot(time, data/data.max(), label=ptname)
            self.data_lines.append(line)
        self.connection.closeTree('spectroscopy', self.shot)

    def remove_vlines(self):
        while self.vlines:
            line = self.vlines.pop()
            line.remove()
            del line

    def remove_data_lines_and_legend(self):
        while self.data_lines:
            line = self.data_lines.pop()
            if isinstance(line, list):
                line = line[0]
            line.remove()
            del line
        for axes in self.axes:
            legend = axes.get_legend()
            if legend:
                legend.remove()
                del legend

    def set_status(self):
        for i,time in enumerate(self.time_markers):
            if time is None:
                self.status_box.set_val(f'Select {self.marker_label[i]}')
                break
            if i == 3:
                self.status_box.set_val(f'Accept markers?')

    def axis_click(self, event):
        if event.inaxes not in self.axes:
            return
        for i in range(4):
            if self.time_markers[i] is None:
                if i >= 1 and event.xdata <= self.time_markers[i-1]:
                    break
                self.time_markers[i] = event.xdata
                for axes in self.axes:
                    plt.sca(axes)
                    self.vlines.append(plt.axvline(event.xdata, color='k'))
                self.set_status()
                if i == 3:
                    self.accept_button.color = 'limegreen'
                    self.accept_button.hovercolor = 'forestgreen'
                break
        plt.draw()


ElmTaggerGUI(save_pdf=True)

plt.show()