#!/bin/bash
#SBATCH -p medium -t 0-4
#SBATCH -N1 -n4 --mem=16G
#SBATCH --mail-user=drsmith@ppppl.gov --mail-type=ALL

date
module -l list

job_label=smithdr_$SLURM_JOB_ID
cd /local-scratch/
mkdir $job_label
cd $job_label
pwd -P

export PYTHONPATH=${HOME}/edge-ml:${PYTHONPATH}
python_exec=/fusion/projects/diagnostics/bes/smithdr/conda/envs/py38/bin/python
python_cmd="from package_bes_data import *; package_bes_data(shots=big_shotlist)"
$python_exec -c $python_cmd &> python_output.txt
python_exit=$?

if [[ $python_exit -eq 0 ]]; then
  mkdir -p ${HOME}/edge-ml/jobs
  mv $job_label ${HOME}/edge-ml/jobs
fi

date

exit $python_exit