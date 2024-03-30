#!/usr/bin/env bash
#SBATCH -t 0-12 -N1 -n8 --mem=60G

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

from elm_data_stats import ELM_Data_Stats

file = '/home/smithdr/ml/elm_data/step_6_labeled_elm_data/elm_data_v1.hdf5'
h5 = ELM_Data_Stats(file)
h5.plot_elms(
    max_elms=None,
    save=True,
)

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
