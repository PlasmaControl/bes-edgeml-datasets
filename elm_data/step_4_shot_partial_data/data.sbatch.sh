#!/usr/bin/env bash
#SBATCH -t 0-8 -N1 -n16 --mem=100G

date

. /home/smithdr/sysenv/omega/bashrc.sh

# prepare module environment
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}

conda activate py311
conda info -e
module list
echo $PYTHONPATH

PYTHON_SCRIPT=$(cat << END

from elm_data.data_tools import HDF5_Data

dataset = HDF5_Data(
    hdf5_file='data_v4.hdf5',
)
dataset.load_shotlist(
    csv_file='/home/smithdr/ml/elm_data/step_3_filter_metadata/filtered_shotlist.csv',
    channels=[21,23],
    with_other_signals=True,
    truncate_hdf5=True, 
    use_concurrent=True, 
)
dataset.print_hdf5_summary()

dataset.plot_8x8_rz_avg()
dataset.plot_ip_bt_histograms()
dataset.plot_configurations()

END
)

# do work
python -c "${PYTHON_SCRIPT}"
python_exit=$?
echo "Python exit status: ${python_exit}"

date

exit $python_exit
