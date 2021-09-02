#!/usr/bin/env bash
#SBATCH -t 0-4
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
cp *shotlist.csv $job_dir
cd $job_dir
pwd -P

# do work
##python ${SLURM_SUBMIT_DIR}/get_metadata.py &> get_metadata.txt
python - <<'HEREDOC' &> get_metadata.txt
from bes_data_tools.bes2hdf5 import package_bes
package_bes(
    shotlist_csvfile='lh_shotlist.csv',
    output_h5file='lh_metadata.hdf5',
    verbose=True)
package_bes(
    shotlist_csvfile='qh_shotlist.csv',
    output_h5file='qh_metadata.hdf5',
    verbose=True)
HEREDOC
python_exit=$?
echo "Python exit status: ${python_exit}"

# move work from local scratch to submission area
mv *metadata.hdf5 ${SLURM_SUBMIT_DIR}
mv get_metadata.txt ${SLURM_SUBMIT_DIR}

date

exit $python_exit
