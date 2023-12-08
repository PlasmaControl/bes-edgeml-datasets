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
    hdf5_file='metadata_v6.hdf5',
)
dataset.load_shotlist(
    csv_file='/home/smithdr/ml/elm_data/step_1_shotlist_from_D3DRDB/shotlist.csv',
    truncate_hdf5=True, 
    use_concurrent=True, 
    only_standard_8x8 = True,
    only_pos_ip = True,
    only_neg_bt = True,
    min_pinj_15l = 500e3,  # power in W
    min_sustained_15l = 300.,  # time in ms
)
dataset.print_hdf5_contents()
dataset.print_hdf5_summary()
dataset.plot_8x8_rz_avg()
dataset.plot_ip_bt_histograms()
dataset.plot_configurations()

END
)

echo ${PYTHON_SCRIPT}

# do work
python -c "${PYTHON_SCRIPT}"
python_exit=$?
echo "Python exit status: ${python_exit}"

date

exit $python_exit
