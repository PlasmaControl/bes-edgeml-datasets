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

PYTHON_SCRIPT=./get_metadata.py
echo -e "\n********** Begin $PYTHON_SCRIPT **********\n"
cat ${PYTHON_SCRIPT}
echo -e "\n********** End $PYTHON_SCRIPT **********\n"

# main
python ${PYTHON_SCRIPT}

python_exit=$?
echo "Python exit status: ${python_exit}"

date

exit $python_exit
