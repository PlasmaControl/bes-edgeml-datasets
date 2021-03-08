#!/usr/bin/env bash
#SBATCH -t 0-6
#SBATCH -N1 -n8 --mem=16G

module load edgeml

. /fusion/projects/diagnostics/bes/smithdr/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd $SLURM_SUBMIT_DIR

python elms.py &> pyout.txt