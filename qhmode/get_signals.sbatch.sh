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

# prepare job dir in local scratch and copy inputs
job_dir=/local-scratch/job_${SLURM_JOB_ID}
mkdir $job_dir
cp *metadata.hdf5 $job_dir

# move to job dir in local scratch
cd $job_dir
pwd -P

# do work
##python ${SLURM_SUBMIT_DIR}/get_signals.py &> get_signals.txt
python - <<'HEREDOC' &> get_signals.txt
from bes_data_tools.package import package_signals_8x8_only
package_signals_8x8_only(
    input_h5file='lh_metadata.hdf5',
    output_h5file='lh_metadata_8x8.hdf5')
package_signals_8x8_only(
    input_h5file='qh_metadata.hdf5',
    output_h5file='qh_metadata_8x8.hdf5')
HEREDOC
python_exit=$?
echo "Python exit status: ${python_exit}"

# move results from local scratch to submission dir
mv bes_signals*.hdf5 ${SLURM_SUBMIT_DIR}
mv *metadata_8x8.hdf5 ${SLURM_SUBMIT_DIR}
mv get_signals.txt ${SLURM_SUBMIT_DIR}

date

exit $python_exit
