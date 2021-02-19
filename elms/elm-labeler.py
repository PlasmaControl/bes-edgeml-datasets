import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib import patches
import MDSplus
import time as timelib


class ElmTaggerGUI(object):

    def __init__(self):
        self.time_markers = [None, None, None, None]
        self.marker_label = ['Pre-ELM start',
                             'ELM start',
                             'ELM end',
                             'Post-ELM end']
        self.fig, self.axes = plt.subplots(ncols=3, figsize=[11.5, 4])
        for axes in self.axes:
            axes.set_xlabel('Time (ms)')
        self.axes[0].set_ylabel('Line avg dens (AU)')
        self.axes[1].set_ylabel('D alpha')
        self.axes[2].set_ylabel('BES')
        plt.tight_layout(rect=(0, 0.15, 1, 1))
        self.fig.canvas.mpl_connect('button_press_event', self.axis_click)

        self.skip_button = widgets.Button(plt.axes([0.03, 0.05, 0.08, 0.075]),
                                          'Skip ELM',
                                          color='lightsalmon',
                                          hovercolor='red')
        self.skip_button.on_clicked(self.skip)

        self.clear_button = widgets.Button(plt.axes([0.15, 0.05, 0.2, 0.075]),
                                           'Clear markers',
                                           color='lightgray',
                                           hovercolor='darkgray')
        self.clear_button.on_clicked(self.clear_markers)

        self.status_box = widgets.TextBox(plt.axes([0.45, 0.05, 0.25, 0.075]),
                                         'Status:',
                                          color='w',
                                          hovercolor='w')

        self.accept_button = widgets.Button(plt.axes([0.75, 0.05, 0.2, 0.075]),
                                            'Accept markers',
                                            color='whitesmoke',
                                            hovercolor='whitesmoke')
        self.accept_button.on_clicked(self.accept)

        self.multi = widgets.MultiCursor(self.fig.canvas, self.axes, color='r', lw=1)

        dtypes = [('ELM index', np.uint16),
                    ('Shot', np.uint32),
                    ('Start time', np.float),
                    ('Stop time', np.float)]
        self.elm_list = np.loadtxt('omfit-elm-time-list.csv',
                                   dtype=dtypes,
                                   delimiter=',')
        self.nelms = self.elm_list['Shot'].size
        self.rng = np.random.default_rng()
        self.processed_elms = []
        self.elm_index = None
        self.shot = None
        self.start_time = None
        self.stop_time = None
        self.connection = MDSplus.Connection('atlas.gat.com')

        self.vlines = []
        self.data_lines = []
        self.clear_and_get_new_elm()

    def skip(self, event):
        # log ELM index, then clear and get new ELM
        self.processed_elms.append(self.elm_index)
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
        # log ELM index, log markers, then clear and get new ELM
        self.processed_elms.append(self.elm_index)
        self.log_elm_markers()
        self.clear_and_get_new_elm()
        plt.draw()

    def log_elm_markers(self):
        pass

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
        while True:
            candidate_index = self.rng.integers(0, self.nelms)
            if candidate_index in self.processed_elms:
                continue
            else:
                self.elm_index = candidate_index
                break
        self.shot = self.elm_list['Shot'][self.elm_index]
        self.start_time = self.elm_list['Start time'][self.elm_index]
        self.stop_time = self.elm_list['Stop time'][self.elm_index]
        self.xlim = [self.start_time - 2*(self.stop_time-self.start_time),
                     self.stop_time + 2*(self.stop_time-self.start_time)]
        print(f'ELM index {self.elm_index}  shot {self.shot}')
        for axes in self.axes:
            axes.set_prop_cycle(None)
        self.plot_density()
        self.plot_bes()
        self.plot_dalpha()
        for axes in self.axes:
            ylim = axes.get_ylim()
            rect = patches.Rectangle([self.start_time, ylim[0]],
                                     self.stop_time-self.start_time,
                                     ylim[1]-ylim[0],
                                     linewidth=0,
                                     facecolor='whitesmoke')
            axes.add_patch(rect)
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
                mask = np.logical_and(time >= self.xlim[0],
                                      time <= self.xlim[1])
                data = data[mask]
                time = time[mask]
                decimate = data.size // 1500 + 1
                data = data[::decimate]
                time = time[::decimate]
                line = plt.plot(time, data/data.max(), label=data_node)
                self.data_lines.append(line)
                break
        plt.xlim(self.xlim)
        # plt.autoscale(enable=True, axis='y')
        plt.ylim(0.7,1.1)
        plt.legend(loc='upper right')
        self.connection.closeTree('electrons', self.shot)

    def plot_bes(self):
        ptnames = ['besfu20', 'besfu23']
        plt.sca(self.axes[2])
        for ptname in ptnames:
            data = None
            time = None
            data_tag = f'ptdata("{ptname}", {self.shot})'
            try:
                t1 = timelib.time()
                data = np.array(self.connection.get(data_tag))
                time = np.array(self.connection.get(f'dim_of({data_tag})'))
                t2 = timelib.time()
                print(f'  Elapsed time for {data_tag}: {t2 - t1:.1f} s')
            except:
                pass
            if data is None or time is None:
                continue
            mask = np.logical_and(time >= self.xlim[0],
                                  time <= self.xlim[1])
            data = data[mask]
            time = time[mask]
            decimate = data.size // 1500 + 1
            data = data[::decimate]
            time = time[::decimate]
            line = plt.plot(time, data/data.max(), label=ptname)
            self.data_lines.append(line)
        plt.xlim(self.xlim)
        plt.autoscale(enable=True, axis='y')
        plt.legend(loc='upper right')

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
            mask = np.logical_and(time >= self.xlim[0],
                                  time <= self.xlim[1])
            data = data[mask]
            time = time[mask]
            decimate = data.size // 1500 + 1
            data = data[::decimate]
            time = time[::decimate]
            line = plt.plot(time, data/data.max(), label=ptname)
            self.data_lines.append(line)
        plt.xlim(self.xlim)
        plt.autoscale(enable=True, axis='y')
        plt.legend(loc='upper right')
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


ElmTaggerGUI()

plt.show()