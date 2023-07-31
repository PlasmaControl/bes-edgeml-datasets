"""
BES_Data class

Fetches and stores BES metadata, relevant signals, and (optionally) BES signals
"""

import time
import numpy as np
import MDSplus


class BES_Data(object):
    _points = ['ip',
               'bt',
               'pinj',
               'pinj_15l',
               'pinj_15r',
               ]

    def __init__(
            self,
            shot=196491,
            channels=None,
            verbose=False,
            get_signals=False,
            only_8x8=True,
            only_standard_8x8=True,
        ):
        t1 = time.time()
        self.connection = MDSplus.Connection('atlas.gat.com')
        if channels is None:
            channels = np.arange(1, 65)
        channels = np.array(channels, dtype=int)
        self.shot = shot
        self.channels = channels
        self.verbose = verbose
        self.time = None
        self.n_time = None
        self.signals = None
        self.metadata = None
        only_8x8 = only_8x8 or only_standard_8x8
        # get time array
        ptdata = f'ptdata("bessu01", {self.shot})'
        try:
            sigtime = self.connection.get(f'dim_of({ptdata})')
            np.array(sigtime).round(4)
        except:
            print(f'{self.shot}: ERROR no BES slow time data')
            self.time = None
            return
        ptdata = f'ptdata("besfu01", {self.shot})'
        try:
            sigtime = self.connection.get(f'dim_of({ptdata})')
            self.time = np.array(sigtime).round(4)
        except:
            self.time = None
            print(f'{self.shot}: ERROR no BES fast time data')
            return
        n_time = self.connection.get(f'size({ptdata})')
        self.n_time = n_time.data()
        assert (self.n_time == self.time.size)
        try:
            # get metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r')).round(2)
            z_position = -np.array(self.connection.get(r'\bes_z')).round(2)
            assert r_position.size == 64 and z_position.size == 64
            self.connection.closeTree('bes', self.shot)
        except:
            print(f'{self.shot}: ERROR getting BES position metadata')
            self.time = None
            return
        is_8x8 = is_standard_8x8 = False
        for i in np.arange(8):
            # del-r of column i
            rdiff = np.diff(r_position[i + np.arange(8) * 8])
            col_test = np.all(np.abs(rdiff) <= 0.13)
            # del-z of row i
            zdiff = np.diff(z_position[i * 8 + np.arange(8)])
            row_test = np.all(np.abs(zdiff) <= 0.13)
            is_8x8 = col_test and row_test
        if is_8x8:
            z_first_column = z_position[np.arange(8) * 8]
            is_standard_8x8 = (z_first_column.argmax() * 8 == 0) and \
                              (z_first_column.argmin() * 8 == 56)
        if only_8x8 and not is_8x8:
            self.time = None
            print(f'{self.shot}: ERROR not 8x8 config')
            return
        if only_standard_8x8 and not is_standard_8x8:
            print(f'{self.shot}: ERROR not standard 8x8 config')
            self.time = None
            return
        self.metadata = {
            'shot': self.shot,
            'delta_time': np.diff(self.time[0:100]).mean().round(4),
            'start_time': self.time[0],
            'stop_time': self.time[-1],
            'n_time': self.n_time,
            'time_units': 'ms',
            'r_position': r_position,
            'z_position': z_position,
            'rz_units': 'cm',
            'date': '',
            'ip': 0.,
            'bt': 0.,
            'is_8x8': is_8x8,
            'is_standard_8x8': is_standard_8x8,
            'r_avg': np.mean(r_position).round(1) if is_8x8 else 0.,
            'z_avg': np.mean(z_position).round(1) if is_8x8 else 0.,
        }
        # get ip, beams, etc.
        for point_name in self._points:
            try:
                if point_name.startswith('pinj'):
                    self.connection.openTree('nb', self.shot)
                    data = np.array(self.connection.get(f'\\{point_name}'))[::2]
                    data_time = np.array(
                            self.connection.get(f'dim_of(\\{point_name})'))[::2]
                    if point_name == 'pinj':
                        date = self.connection.get(
                            f'getnci(\\{point_name}, "time_inserted")')
                        self.metadata['date'] = date.date.decode('utf-8')
                    self.connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(self.connection.get(ptdata))[::8]
                    data_time = np.array(self.connection.get('dim_of(_n)'))[::8]
                time_mask = np.logical_and(data_time >= self.time[0],
                                           data_time <= self.time[-1])
                data = data[time_mask]
                data_time = data_time[time_mask]
                if point_name == 'bt':
                    print(f"Bt avg/std: {data.mean():.6f} {np.std(data):.6f}")
            except:
                self.time = None
                print(f'{self.shot}: ERROR for data node {point_name}')
                return
            assert data.shape == data_time.shape
            setattr(self, point_name, data)
            if point_name == 'pinj' or 'pinj' not in point_name:
                setattr(self, f'{point_name}_time', data_time)
        if self.pinj_15l.max() < 500e3:
            self.time = None
            print(f'{self.shot}: ERROR small pinj_15l')
            return
        for ptname in ['ip', 'bt']:
            try:
                data: np.ndarray = getattr(self, ptname)
                if np.abs(data.max()) >= np.abs(data.min()):
                    self.metadata[ptname] = data.max()
                else:
                    self.metadata[ptname] = data.min()
            except:
                self.time = None
                print(f'{self.shot}: ERROR {ptname}')
                return
        print(f'{self.shot}: {self.n_time} time points')
        print(f'{self.shot}: Metadata time = {time.time() - t1:.2f} s')
        if get_signals:
            self.get_signals()

    def get_signals(
        self,
        channels=None,
    ):
        assert self.time is not None
        if channels:
            self.channels = np.array(channels, dtype=int)
        t1 = time.time()
        print(f'{self.shot}: fetching {self.channels.size} signals')
        tdi_vars = []
        tdi_assignments = []
        for channel in self.channels:
            var = f'_n{channel:02d}_{self.shot}'
            tdi_vars.append(var)
            tmp = f'{var} = ptdata("besfu{channel:02d}", {self.shot})'
            tdi_assignments.append(tmp)
        self.signals = np.empty([self.channels.size, self.n_time])
        try:
            self.connection.get(', '.join(tdi_assignments))
            for i, tdi_var in enumerate(tdi_vars):
                self.signals[i, :] = self.connection.get(tdi_var)
        except:
            print(f'{self.shot}: ERROR fetching signals')
            self.time = None
            self.signals = None
            return
        print(f'{self.shot}: Signal time = {time.time() - t1:.2f} s')


if __name__=='__main__':
    bes_data = BES_Data(
        # shot=184800,
        # channels=[1,2,3,4],
        # get_signals=True,
        # verbose=True,
    )