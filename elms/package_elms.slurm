#!/usr/bin/env bash
#SBATCH -p medium -t 0-8
#SBATCH -N1 -n16 --mem=32G

date
module load edgeml
export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}
module -l list
. /home/smithdr/packages/miniconda3/etc/profile.d/conda.sh
conda activate py38

# make work area in local scratch
job_label="job_${SLURM_JOB_ID}"
cd /local-scratch
mkdir $job_label
cd $job_label
pwd -P

# do work
python_exec=/fusion/projects/diagnostics/bes/smithdr/conda/envs/py38/bin/python
##$python_exec -c "import elms; elms.package_elms()" &> python.txt
$python_exec -c "import elms; elms.package_8x8_sublist(max_shots=12)" &> python.txt
python_exit=$?
echo "Python exit status: ${python_exit}"

# move work to project area
cd $SLURM_SUBMIT_DIR
pwd -P
mkdir -p ${SLURM_SUBMIT_DIR}/data
mv /local-scratch/${job_label} ${SLURM_SUBMIT_DIR}/data

date

exit $python_exit