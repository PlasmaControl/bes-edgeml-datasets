#!/usr/bin/env bash
#SBATCH -t 0-8 -N1 -n16 --mem=100G

start_time=`date`
echo Start time: $start_time

. /home/smithdr/sysenv/omega/bashrc.sh

# prepare module environment
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}

conda activate py311
conda info -e
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
