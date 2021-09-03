"""
BES_Data class

Fetches and stores BES metadata, relevant signals, and (optionally) BES signals
"""

from pathlib import Path
import time
import numpy as np
import h5py
import MDSplus


# make standard directories
Path('data').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)


class BES_Data(object):
    _points = ['ip',
               'bt',
               'pinj',
               'pinj_15l',
               'vinj_15l',
               'pinj_15r',
               'vinj_15r',
               ]

    def __init__(self,
                 shot=None,
                 channels=None,
                 verbose=False,
                 get_signals=False):
        t1 = time.time()
        self.connection = MDSplus.Connection('atlas.gat.com')
        if shot is None:
            shot = 176778
        if channels is None:
            channels = np.arange(1, 65)
        channels = np.array(channels)
        self.shot = shot
        self.channels = channels
        self.verbose = verbose
        self.time = None
        self.signals = None
        self.metadata = None
        print(f'{self.shot}: start')
        # get time array
        ptdata = f'ptdata("besfu01", {self.shot})'
        try:
            sigtime = self.connection.get(f'dim_of({ptdata})')
            self.time = np.array(sigtime).round(4)
        except:
            self.time = None
            print(f'{self.shot}: ERROR no time data')
            return
        n_time = self.connection.get(f'size({ptdata})')
        self.n_time = n_time.data()
        assert (self.n_time == self.time.size)
        try:
            # get metadata
            self.connection.openTree('bes', self.shot)
            r_position = np.array(self.connection.get(r'\bes_r')).round(2)
            z_position = np.array(self.connection.get(r'\bes_z')).round(2)
            start_time = self.connection.get(r'\bes_ts')
            self.connection.closeTree('bes', self.shot)
        except:
            print(f'{self.shot}: ERROR getting metadata')
            self.time = None
            return
        if not start_time == self.time[0]:
            print(f'{self.shot}: ALERT inconsistent start times: ',
                  start_time, self.time[0])
        self.metadata = {'shot': self.shot,
                         'delta_time': np.diff(self.time[0:100]).mean().round(
                             4),
                         'start_time': self.time[0],
                         'stop_time': self.time[-1],
                         'n_time': self.n_time,
                         'time_units': 'ms',
                         'r_position': r_position,
                         'z_position': z_position,
                         'rz_units': 'cm',
                         'date': ''}
        # get ip, beams, etc.
        for point_name in self._points:
            try:
                if 'inj' in point_name:
                    self.connection.openTree('nb', self.shot)
                    data = np.array(self.connection.get(f'\\{point_name}'))
                    data_time = np.array(
                            self.connection.get(f'dim_of(\\{point_name})'))
                    if point_name == 'pinj':
                        date = self.connection.get(
                            f'getnci(\\{point_name}, "time_inserted")')
                        self.metadata['date'] = date.date.decode('utf-8')
                    self.connection.closeTree('nb', self.shot)
                else:
                    ptdata = f'_n = ptdata("{point_name}", {self.shot})'
                    data = np.array(self.connection.get(ptdata))
                    data_time = np.array(self.connection.get('dim_of(_n)'))
                time_mask = np.logical_and(data_time >= self.time[0],
                                           data_time <= self.time[-1])
                data = data[time_mask]
                data_time = data_time[time_mask]
            except:
                if point_name == 'pinj_15l':
                    self.time = None
                    print(f'{self.shot}: ERROR missing pinj_15l')
                    return
                print(f'{self.shot}: INVALID data node for {point_name}')
                data = h5py.Empty(dtype='f')
                data_time = h5py.Empty(dtype='f')
            assert (data.shape == data_time.shape)
            setattr(self, point_name, data)
            if point_name == 'pinj' or 'inj' not in point_name:
                setattr(self, f'{point_name}_time', data_time)
            if point_name =='pinj_15l':
                if data.max() < 500e3:
                    self.time = None
                    print(f'{self.shot}: ERROR invalid pinj_15l')
                    return
        print(f'{self.shot}: {self.n_time} time points')
        t2 = time.time()
        print(f'{self.shot}: Metadata time = {t2 - t1:.2f} s')
        if get_signals:
            self.get_signals()

    def get_signals(self):
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
        t2 = time.time()
        print(f'{self.shot}: Signal time = {t2 - t1:.2f} s')


if __name__=='__main__':
    bes_data = BES_Data(shot=184800,
                        channels=[1,2,3,4],
                        get_signals=True,
                        verbose=True)