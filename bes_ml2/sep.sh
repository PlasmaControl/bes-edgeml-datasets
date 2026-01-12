#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=12
#SBATCH --time=12:30:00
#SBATCH --qos=regular
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
import time

import numpy as np

from bes_ml2.main import BES_Trainer
from bes_ml2 import separatrix_datamodule
from bes_ml2 import elm_lightning_model

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
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

    n_rows = 6
    n_cols = 8

    datamodule = separatrix_datamodule.Separatrix_Datamodule(
            data_file = '/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20241014.hdf5',
            # max_shots = 10,
            signal_window_size=1,
            n_rows = n_rows,
            n_cols = n_cols,
            fraction_test=0.05,
            fraction_validation=0.15,
            batch_size=512,
            plot_data_stats=False,
            num_workers=4,
            seed=0,
            world_size=world_size,
            lower_cutoff_frequency_hz=15e3,
            upper_cutoff_frequency_hz=150e3,
        )

    weight_decay = 1e-4
    cnn_nlayers = 1
    trial_name = f'{logger_hash}'

    lightning_model = elm_lightning_model.Lightning_Model(
        encoder_lr=1e-4,
        decoder_lr=1e-5,
        signal_window_size=datamodule.signal_window_size,
        n_rows=datamodule.n_rows,
        n_cols=datamodule.n_cols,
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=10,
        weight_decay=weight_decay,
        cnn_dropout=0.05,
        encoder_type='none',
        cnn_nlayers=cnn_nlayers,
        cnn_num_kernels=(10),
        cnn_kernel_time_size=[2] * cnn_nlayers,
        cnn_kernel_spatial_size=3,
        cnn_maxpool_time_size=(2),
        cnn_maxpool_spatial_size=[(1, 1)],
        leaky_relu_slope=0.001,
        mlp_layers=(50, 50),
        mlp_dropout=0.1,
        separatrix_mlp=True,
        velocimetry_mlp=False,
        multiclass_classifier_mlp=False,
        reconstruction_decoder=False,
        time_to_elm_mlp=False,
        classifier_mlp=False,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
        # num_train_batches=1,
        # num_val_batches=1,
        # val_check_interval=1,
        num_train_batches=2000, 
        val_check_interval=1000,
    )

    trainer.run_all(
        max_epochs=100,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=30,
        # skip_test=True,
        # skip_predict=True,
        # debug_predict=True,
    )
    print(f'Python elapsed time {(time.time()-t_start)/60:.1f} min')
except:
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