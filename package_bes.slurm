#!/bin/bash
#SBATCH -p medium -t 0-4
#SBATCH -N1 -n4 --mem=16G
#SBATCH --mail-user=drsmith@ppppl.gov --mail-type=ALL

date
module -l list

cd $SLURM_SUBMIT_DIR

#job_label=smithdr_$SLURM_JOB_ID
#cd /local-scratch/
#mkdir $job_label
#cd $job_label
#pwd -P

#export PYTHONPATH=${HOME}/edge-ml:${PYTHONPATH}
python_exec=/fusion/projects/diagnostics/bes/smithdr/conda/envs/py38/bin/python
#python_cmd="from package_bes_data import *; package_bes_data(shots=big_shotlist)"
$python_exec -c "from package_bes_data import *;package_bes_data(shots=big_shotlist, verbose=True, with_signals=True)" &> python.txt

#mkdir -p ${HOME}/edge-ml/jobs
#mv $job_label ${HOME}/edge-ml/jobs

date

exit

