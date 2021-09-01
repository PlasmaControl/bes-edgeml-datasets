#!/usr/bin/env bash
#SBATCH -t 1-0
#SBATCH -N1 -n8 --mem=16G

date

# prepare module environment
module load edgeml
## export PYTHONPATH=${SLURM_SUBMIT_DIR}:${PYTHONPATH}
module -l list

# activate conda environment
. /fusion/projects/diagnostics/bes/smithdr/miniconda3/etc/profile.d/conda.sh
conda activate py38
conda info -e

# prepare work area in local scratch
job_dir=/local-scratch/job_${SLURM_JOB_ID}
mkdir $job_dir
cp *metadata.hdf5 $job_dir
cd $job_dir
pwd -P

# do work
python $HOME/edgeml/qhmode/get_signals.py &> get_signals.txt
python_exit=$?
echo "Python exit status: ${python_exit}"

# move work from local scratch to submission area
mv bes_signals*.hdf5 ${SLURM_SUBMIT_DIR}
mv *metadata_8x8.hdf5 ${SLURM_SUBMIT_DIR}
mv get_signals.txt ${SLURM_SUBMIT_DIR}

date

exit $python_exit
