import time as timelib
from pathlib import Path
import subprocess
import shutil
from typing import Sequence, Union


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
import MDSplus
import h5py

from bes_data_tools import package_h5


class ElmTaggerGUI(object):

    def __init__(
        self,
        input_unlabeled_elm_event_file=None,
        output_labeled_elm_event_filename=None,
        save_pdf=True,
        manual_elm_list=None,
        merge_pdfs_on_close=True,
    ):
        self.manual_elm_list = manual_elm_list
        self.time_markers = [None, None, None, None]
        self.marker_label = ['Pre-ELM start',
                             'ELM start',
                             'ELM end',
                             'Post-ELM end']
        self.fig, self.axes = plt.subplots(nrows=3, figsize=[15, 8])
        # self.fig_num = self.fig.number
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        for axes in self.axes:
            axes.set_xlabel('Time (ms)')
        self.axes[0].set_ylabel('Line avg dens (AU)')
        self.axes[1].set_ylabel('D alpha')
        self.axes[2].set_ylabel('BES')
        plt.suptitle('empty')
        plt.tight_layout(rect=(0, 0.08, 1, 1))

        self.fig.canvas.mpl_connect('button_press_event', self.axis_click)

        self.save_pdf = save_pdf

        self.skip_button = widgets.Button(plt.axes([0.01, 0.02, 0.07, 0.04]),
                                          'Skip ELM',
                                          color='lightsalmon',
                                          hovercolor='red')
        self.skip_button.on_clicked(self.skip)

        self.clear_button = widgets.Button(plt.axes([0.2, 0.02, 0.2, 0.04]),
                                           'Clear markers',
                                           color='lightgray',
                                           hovercolor='darkgray')
        self.clear_button.on_clicked(self.clear_markers)

        self.status_box = widgets.TextBox(plt.axes([0.5, 0.02, 0.2, 0.04]),
                                          'Status:',
                                          color='w',
                                          hovercolor='w')

        self.accept_button = widgets.Button(plt.axes([0.75, 0.02, 0.2, 0.04]),
                                            'Accept markers',
                                            color='whitesmoke',
                                            hovercolor='whitesmoke')
        self.accept_button.on_clicked(self.accept)

        self.multi = widgets.MultiCursor(self.fig.canvas, self.axes, color='r', lw=1)

        assert Path(input_unlabeled_elm_event_file).exists()
        self.unlabeled_elm_events_h5 = h5py.File(input_unlabeled_elm_event_file, 'r')
        self.candidate_elms = np.array([int(elm_key) for elm_key in self.unlabeled_elm_events_h5], dtype=int)
        self.n_elms = len(self.unlabeled_elm_events_h5)
        self.last_index = self.candidate_elms.max()
        # shots = np.unique([elm_event.attrs['shot'] for elm_event in self.unlabeled_elm_events_h5.values()])
        print(f'Unlabeled ELM event data file: {input_unlabeled_elm_event_file}')
        print(f'  Number of ELM events: {self.n_elms}')
        print(f'  Last ELM index: {self.last_index}')
        # print(f'  Number of unique shots: {shots.size}')

        output_file = Path().resolve() / output_labeled_elm_event_filename
        print(f'Output labeled ELM event file: {output_file}')
        print(f'  File exists: {output_file.exists()}')
        self.labeled_elm_events_h5 = h5py.File(output_file.as_posix(), 'a')

        self.figures_dir = Path().resolve() / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

        self.labeled_elms = self.labeled_elm_events_h5.attrs.setdefault('labeled_elms',
                                                             np.array([], dtype=int))
        self.skipped_elms = self.labeled_elm_events_h5.attrs.setdefault('skipped_elms',
                                                             np.array([], dtype=int))

        self.validate_data_file()

        self.rng = np.random.default_rng()

        self.remaining_candidate_elms = np.setxor1d(self.candidate_elms, self.labeled_elms, assume_unique=True)
        self.remaining_candidate_elms = np.setxor1d(self.remaining_candidate_elms, self.skipped_elms, assume_unique=True)
        self.rng.shuffle(self.remaining_candidate_elms)
        print(f"Remaining candidate ELMs: {self.remaining_candidate_elms.size}")
        # assert (self.remaining_candidate_elms.size + self.labeled_elms.size + self.skipped_elms.size) == self.candidate_elms.size

        self.elm_index = None
        self.shot = None
        self.start_time = None
        self.stop_time = None
        self.time = None
        self.signals = None
        self.connection = MDSplus.Connection('atlas.gat.com')
        self.merge_pdfs_on_close = merge_pdfs_on_close

        self.vlines = []
        self.data_lines = []
        # plt.show()
        self.clear_and_get_new_elm()
        plt.show()

    def validate_data_file(self):
        if len(self.labeled_elm_events_h5) != self.labeled_elms.size:
            self.labeled_elms = np.array([int(elm_index) for elm_index in self.labeled_elm_events_h5],
                                         dtype=int)
            self.labeled_elm_events_h5.attrs['labeled_elms'] = self.labeled_elms
        for elm_index in self.labeled_elm_events_h5:
            assert int(elm_index) in self.labeled_elms
            assert int(elm_index) not in self.skipped_elms
        print('Labeled data file is valid')

    def print_progress_summary(self):
        print(f"Skipped ELMs: {self.skipped_elms.size}")
        print(f"Labeled ELMs: {self.labeled_elms.size}")
        print(f"Remaining candidate ELMs: {self.remaining_candidate_elms.size}")
        shots, counts = np.unique(
                [elm_event.attrs['shot'] for elm_event in self.labeled_elm_events_h5.values()],
                return_counts=True,
        )
        print(f"Unique shots: {shots.size}")
        print(f"Max ELMs for single shot: {counts.max()} ({np.count_nonzero(counts==counts.max())} shots with max)")

    def on_close(self, event):
        print('on_close')
        self.labeled_elm_events_h5.attrs['labeled_elms'] = self.labeled_elms
        self.labeled_elm_events_h5.attrs['skipped_elms'] = self.skipped_elms
        package_h5.print_h5py_contents(self.labeled_elm_events_h5.filename)
        self.labeled_elm_events_h5.flush()
        self.validate_data_file()
        self.print_progress_summary()
        if self.labeled_elm_events_h5:
            self.labeled_elm_events_h5.close()
        if self.unlabeled_elm_events_h5:
            self.unlabeled_elm_events_h5.close()
        if self.merge_pdfs_on_close:
            merge_pdfs()

    def skip(self, event):
        # log ELM index, then clear and get new ELM
        self.skipped_elms = np.append(self.skipped_elms, self.elm_index)
        self.labeled_elm_events_h5.attrs['skipped_elms'] = self.skipped_elms
        self.labeled_elm_events_h5.flush()
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
            pdf_file = self.figures_dir / \
                       f'elm_{self.elm_index:05d}_shot_{self.shot}.pdf'
            plt.savefig(pdf_file, format='pdf', transparent=True)
        self.labeled_elms = np.append(self.labeled_elms, self.elm_index)
        self.labeled_elm_events_h5.attrs['labeled_elms'] = self.labeled_elms
        self.log_elm_markers()
        self.labeled_elm_events_h5.flush()
        self.clear_and_get_new_elm()
        plt.draw()

    def log_elm_markers(self):
        time = self.time
        signals = self.signals
        mask = np.logical_and(time >= self.time_markers[0],
                              time <= self.time_markers[3])
        time = time[mask]
        signals = signals[:, mask]
        labels = np.zeros(time.shape, dtype=int8)
        mask = np.logical_and(time >= self.time_markers[1],
                              time <= self.time_markers[2])
        labels[mask] = 1

        groupname = f'{self.elm_index:05d}'
        assert(groupname not in self.labeled_elm_events_h5)
        elm_group = self.labeled_elm_events_h5.create_group(groupname)
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
        # rng_range = self.n_elms - 1
        # attempts = 0
        while True:
            # attempts += 1
            # if self.manual_elm_list:
            #     candidate_index = int(self.manual_elm_list.pop(0))
            # else:
            #     candidate_index = self.rng.integers(0, rng_range+375)
            if self.remaining_candidate_elms.size == 0:
                print("No candidate ELMs remaining, closing")
                self.on_close()
            candidate_index, self.remaining_candidate_elms = \
                self.remaining_candidate_elms[0], self.remaining_candidate_elms[1:]
            if (candidate_index in self.labeled_elms) or \
                    (candidate_index in self.skipped_elms):
                continue
            if f'{candidate_index:05d}' not in self.unlabeled_elm_events_h5:
                continue
            self.elm_index = candidate_index
            # assert((self.elm_index-user_index)%4 == 0)
            elm_group = self.unlabeled_elm_events_h5[f'{self.elm_index:05d}']
            self.signals = elm_group['signals'][:, :]
            max_signal = np.amax(self.signals[[19, 21, 23], :])
            if max_signal < 9.9:
                self.skipped_elms = np.append(self.skipped_elms, self.elm_index)
                continue
            self.shot = elm_group.attrs['shot']
            self.time = elm_group['time'][:]
            if self.time.size != self.signals.shape[1]:
                self.skipped_elms = np.append(self.skipped_elms, self.elm_index)
                continue
            break
        self.start_time = self.time[0]
        self.stop_time = self.time[-1]
        print(f'ELM index {self.elm_index}  shot {self.shot}')
        for axes in self.axes:
            axes.set_prop_cycle(None)
        plt.suptitle(f'Shot {self.shot} | ELM index {self.elm_index}')
        self.plot_density()
        self.plot_bes()
        self.plot_dalpha()
        for axes in self.axes.flat:
            plt.sca(axes)
            plt.legend(loc='upper right')
            plt.xlim(self.start_time, self.stop_time)
        self.labeled_elm_events_h5.flush()
        self.print_progress_summary()
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
        decimate = self.time.size // 1000 + 1
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


def merge_pdfs():
    inputs = sorted(Path().glob('figures/elm*.pdf'))
    output = Path('labeled_elms.pdf')
    gs_cmd = shutil.which('gs')
    if gs_cmd is None:
        return
    if output.exists():
        output.unlink()
    print(f"Number of source files: {len(inputs)}")
    print(f"Merging PDFs into file: {output.as_posix()}  (this can take ~30 s)")
    cmd = [
        gs_cmd,
        '-q',
        '-dBATCH',
        '-dNOPAUSE',
        '-sDEVICE=pdfwrite',
        '-dPDFSETTINGS=/prepress',
        '-dCompatibilityLevel=1.4',
    ]
    cmd.append(f"-sOutputFile={output.as_posix()}")
    for pdf_file in inputs:
        cmd.append(f"{pdf_file.as_posix()}")
    result = subprocess.run(cmd, check=True)
    assert result.returncode == 0 and output.exists()
    print("Success")


if __name__=='__main__':
    ElmTaggerGUI(
        input_unlabeled_elm_event_file=(
                Path().resolve().parent /
                'data/step_5_unlabeled_elm_events_long_windows.hdf5'
        ),
        output_labeled_elm_event_filename='step_6_labeled_elm_events.hdf5',
        save_pdf=True,
        manual_elm_list=None,
        merge_pdfs_on_close=False,
    )
