#!/usr/bin/env bash
#SBATCH -t 1-0 -N1 -n8 --mem=80G

start_time=`date`
echo Start time: $start_time

# prepare module environment
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}

. "/home/smithdr/.miniconda3/etc/profile.d/conda.sh"
conda activate py313
conda info -e
pyinfo
module list
echo $PYTHONPATH

PYTHON_SCRIPT=$(
cat <<'END_HEREDOC'
from elm_data_stats import ELM_Data_Stats

file = '/home/smithdr/ml/data-pipeline/step_6_labeled_elm_data/elm_data_v1.hdf5'
h5 = ELM_Data_Stats(file, save_dir='./figures_4_150_v2')
# h5.plot_elms(
#     save=True,
#     shuffle=True,
#     fir_bp_low=4.,
#     fir_bp_high=150.,
# )
h5.plot_shot_elm_stats(save=True)
h5.plot_channel_stats(
    save=True,
    fir_bp_low=4.,
    fir_bp_high=150.,
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
