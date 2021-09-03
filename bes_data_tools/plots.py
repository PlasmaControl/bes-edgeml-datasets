"""
Plotting utilities for BES metadata and signal HDF5 files.

`plot_layouts` is a function that plots BES channel layouts from metadata file.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import h5py


# make standard directories
Path('data').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)


def plot_layouts(input_h5file=None,
                 batch_and_save=False,
                 output_pdffile='configurations.pdf'):
    with h5py.File(input_h5file, 'r') as hfile:
        print(hfile.filename)
        config_groups = hfile['configurations']['8x8_configurations']
        print(f'{len(config_groups)} 8x8 configurations')
        total_shots = 0
        plt.figure(figsize=(4,4))
        plt.subplot(111)
        for name, config in config_groups.items():
            print(f'Configuration {name}')
            shots = config.attrs['shots']
            total_shots += shots.size
            print(f'  {shots.size} shots: ', shots)
            r_position = config.attrs['r_position']
            z_position = config.attrs['z_position'] * -1
            plt.cla()
            plt.plot(r_position, z_position, 'x')
            for i in range(r_position.size):
                plt.annotate(repr(i+1),
                             (r_position[i], z_position[i]),
                             textcoords='offset points',
                             xytext=(2, 4),
                             size='x-small')
            plt.xlim(218, 232)
            plt.xlabel('R (cm)')
            plt.ylim(-8, 8)
            plt.ylabel('Z (cm)')
            plt.gca().set_aspect('equal')
            plt.title(f'Config {name}')
            plt.tight_layout()
            delz = z_position.max() - z_position.min()
            plt.annotate(f'delz = {delz:.1f} cm',
                         (0.03,0.03),
                         xycoords='axes fraction',
                         size='small')
            if batch_and_save:
                pass
            else:
                print('Keystroke or mouse click to continue...')
                plt.waitforbuttonpress()
        print(f'Total shots: {total_shots}')

if __name__=='__main__':
    plt.close('all')
    h5file = Path.home() / 'edgeml/qhmode/lh_metadata_8x8.hdf5'
    plot_layouts(input_h5file=h5file)