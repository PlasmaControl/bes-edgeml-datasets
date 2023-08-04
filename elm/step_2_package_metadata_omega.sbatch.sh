#!/usr/bin/env bash
#SBATCH -t 0-12 -N1 -n16 --mem=100G

date

. /home/smithdr/sysenv/omega/bashrc.sh

# prepare module environment
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}

conda activate py311
conda info -e
module list
echo $PYTHONPATH

PYTHON_SCRIPT=$(cat << END

from bes_data_tools.bes_data_tools import BES_Metadata

dataset = BES_Metadata(
    hdf5_file='data/big_metadata_v2.hdf5',
)
dataset.load_shotlist(
    csv_file='data/big_shotlist.csv',
    truncate_hdf5=True, 
    use_concurrent=True, 
    only_standard_8x8 = True,
)
dataset.print_hdf5_contents()
dataset.plot_8x8_configurations()

END
)

# do work
python -c "${PYTHON_SCRIPT}"
python_exit=$?
echo "Python exit status: ${python_exit}"

date

exit $python_exit
