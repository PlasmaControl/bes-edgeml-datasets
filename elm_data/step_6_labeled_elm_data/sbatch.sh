#!/usr/bin/env bash
#SBATCH -t 0-8 -N1 -n16 --mem=100G

start_time=`date`
echo Start time: $start_time

# prepare module environment
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}

. "/home/smithdr/.miniconda3/etc/profile.d/conda.sh"
conda activate py311
conda info -e
pyinfo
module list
echo $PYTHONPATH

PYTHON_SCRIPT=./add_elm_data.py
echo -e "\n********** Begin $PYTHON_SCRIPT **********\n"
cat ${PYTHON_SCRIPT}
echo -e "\n********** End $PYTHON_SCRIPT **********\n"

# main
echo Running $PYTHON_SCRIPT ...
python $PYTHON_SCRIPT

python_exit=$?
echo Python exit status: $python_exit

stop_time=`date`

echo Start time: $start_time
echo Stop time: $stop_time

exit $python_exit
