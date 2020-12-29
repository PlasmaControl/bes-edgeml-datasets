#!/bin/bash
#SBATCH -p medium -t 0-4
#SBATCH -N1 -n4 --mem=16G

date
module -l list

# make work area in local scratch
job_label="job_${SLURM_JOB_ID}"
cd /local-scratch
mkdir $job_label
cd $job_label
pwd -P

# do work
export PYTHONPATH=${HOME}/edge-ml:${PYTHONPATH}
python_exec=/fusion/projects/diagnostics/bes/smithdr/conda/envs/py38/bin/python
$python_exec -c "import bes2hdf5; bes2hdf5.small_job()" &> python.txt
python_exit=$?

# move work to project area
cd $SLURM_SUBMIT_DIR
pwd -P
mkdir -p ${SLURM_SUBMIT_DIR}/jobs
mv /local-scratch/${job_label} ${SLURM_SUBMIT_DIR}/jobs

date

exit $python_exit