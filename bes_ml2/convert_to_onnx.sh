#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=4
#SBATCH --time=00:30:00
#SBATCH --qos=debug
###SBATCH --array=0

echo Python executable: $(which python)
echo
echo Job name: $SLURM_JOB_NAME
echo QOS: $SLURM_JOB_QOS
echo Account: $SLURM_JOB_ACCOUNT
echo Submit dir: $SLURM_SUBMIT_DIR
echo
echo Job array ID: $SLURM_ARRAY_JOB_ID
echo Job ID: $SLURM_JOBID
echo Job array task: $SLURM_ARRAY_TASK_ID
echo Job array task count: $SLURM_ARRAY_TASK_COUNT
echo
echo Nodes: $SLURM_NNODES
echo Head node: $SLURMD_NODENAME
echo hostname $(hostname)
echo Nodelist: $SLURM_NODELIST
echo Tasks per node: $SLURM_NTASKS_PER_NODE
echo GPUs per node: $SLURM_GPUS_PER_NODE

if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export UNIQUE_IDENTIFIER=$SLURM_JOBID
fi
echo UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER

JOB_DIR=/pscratch/sd/k/kevinsg/bes_ml_jobs/
mkdir --parents $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

export WANDB__SERVICE_WAIT=300

PYTHON_SCRIPT=$(cat << END

import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta

import numpy as np

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import torch
import wandb

from bes_ml2.main import BES_Trainer
from bes_ml2 import confinement_datamodule_3
from bes_ml2 import elm_lightning_model

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
num_nodes = int(os.getenv('SLURM_NNODES', '1'))  # Default to 1 if not set
world_size = int(os.getenv('SLURM_NTASKS', 0))
world_rank = int(os.getenv('SLURM_PROCID', 0))
local_rank = int(os.getenv('SLURM_LOCALID', 0))
node_rank = int(os.getenv('SLURM_NODEID', 0))
print(f'World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})')

is_global_zero = world_rank == 0

if not is_global_zero:
    f = open(os.devnull, 'w')
    sys.stdout = f

try:
    t_start = time.time()

    checkpoint = '/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/25050889_copy/checkpoints/best.ckpt'

    # load data and model from checkpoint
    lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(checkpoint_path=checkpoint)
    datamodule = confinement_datamodule_3.Confinement_Datamodule.load_from_checkpoint(checkpoint_path=checkpoint)
    datamodule.setup(stage='test')

    float_precision = '16-mixed' if torch.cuda.is_available() else 32
    frontends_active = [value for value in lightning_model.frontends_active.values()]
    some_unused = False in frontends_active

    # Initialize the Trainer
    trainer = Trainer(
        num_nodes=num_nodes,
        precision=float_precision,
        strategy=DDPStrategy(find_unused_parameters=some_unused, timeout=timedelta(seconds=9600)),
    )
    # Ensure the model is on CPU and in evaluation mode
    model = lightning_model.to('cpu')
    model.eval()

    # Generate a dummy input from the datamodule test dataloader
    test_dataloader = datamodule.test_dataloader()
    batch = next(iter(test_dataloader))
    dummy_input = batch[0]  # Assuming the first element of the batch is the input

    # Path to save the ONNX model
    onnx_file_path = '/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/25050889_copy/checkpoints/25050889_model.onnx'

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=17,
                    do_constant_folding=True, input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print(f'Model has been converted to ONNX format and saved to {onnx_file_path}')

    print(f'Python elapsed time {(time.time()-t_start)/60:.1f} min')
except Exception as e:
    print(f"An error occurred: {e}")
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
    raise
finally:
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
END
)

echo Script:
echo "${PYTHON_SCRIPT}"


START_TIME=$(date +%s)
srun python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE