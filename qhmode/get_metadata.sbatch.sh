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
cp data/*shotlist.csv $job_dir
cd $job_dir
pwd -P

python_stdout='get_metadata.out'

# do work
python - <<'HEREDOC' &> ${python_stdout}
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

# move results to submission dir
mv ${python_stdout} ${SLURM_SUBMIT_DIR}
mv *metadata.hdf5 ${SLURM_SUBMIT_DIR}/data/

date

exit $python_exit
