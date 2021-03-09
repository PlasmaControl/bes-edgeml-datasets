#!/usr/bin/env bash


# reset modules
module purge
module load defaults

cd /fusion/projects/diagnostics/bes/smithdr/labeled-elms

# load conda
source /fusion/projects/diagnostics/bes/smithdr/miniconda3/etc/profile.d/conda.sh

# activate conda environment
conda activate /fusion/projects/diagnostics/bes/smithdr/conda/envs/py38

# add edgeml module to PYTHONPATH
export PYTHONPATH=/fusion/projects/diagnostics/bes/smithdr/edgeml:${PYTHONPATH}

# launch labeler app
python /fusion/projects/diagnostics/bes/smithdr/edgeml/elm-labeling-tool/elm-labeler.py
