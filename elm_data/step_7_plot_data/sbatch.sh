#!/usr/bin/env bash
#SBATCH -t 0-6 -N1 -n6 --mem=60G

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

PYTHON_SCRIPT=$(
cat <<'END_HEREDOC'
from plot_data import plot_elms
file = '/home/smithdr/ml/elm_data/step_6_labeled_elm_data/elm_data_v1.hdf5'
plot_elms(file)
END_HEREDOC
)

echo -e "\n********** Begin script **********\n"
echo "$PYTHON_SCRIPT"
echo -e "\n********** End script **********\n"

# main
echo Running script ...
python -c "$PYTHON_SCRIPT"

python_exit=$?
echo Python exit status: $python_exit

stop_time=`date`

echo Start time: $start_time
echo Stop time: $stop_time

exit $python_exit
