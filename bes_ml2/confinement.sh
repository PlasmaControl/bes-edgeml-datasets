#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=2
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
import time

import numpy as np

from bes_ml2.main import BES_Trainer
from bes_ml2 import confinement_datamodule_4
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

    # --- Your selection knobs (must match what you pass to the datamodule) ---
    block_cols   = ('last', 4)
    row_stride   = 1
    row_offset   = 0

    # Compute R_sel, C_sel without needing a datamodule instance
    import numpy as np
    R_sel = len(np.arange(8)[row_offset::row_stride])

    def _cols_from_spec(spec, n=8):
        if isinstance(spec, tuple) and spec[0] == 'last':
            k = int(spec[1]); return np.arange(n-k, n)
        if isinstance(spec, slice):       return np.arange(n)[spec]
        if isinstance(spec, (list, np.ndarray)): return np.array(spec, dtype=int)
        raise ValueError("bad block_cols")

    C_sel = _cols_from_spec(block_cols, 8).size  # -> 4
    label_filter={3,4}
    datamodule = confinement_datamodule_4.Confinement_Datamodule(
        data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20250824_confinement_final.hdf5',
        seed=2,
        # lower_cutoff_frequency_hz=2.5e3,
        # upper_cutoff_frequency_hz=200e3,
        target_sampling_hz=250e3,
        standardize_signals=False,
        signal_window_size=64,
        predict_window_stride=32,
        n_rows=2,
        n_cols=6,
        label_filter=label_filter,
        num_classes=len(label_filter),
        fraction_test=0.22,
        fraction_validation=0.1,
        batch_size=256,
        num_workers=4,
        plot_data_stats=False,
        world_size=world_size,
        one_hot_labels=True,
        # force_validation_shots=['145384', '203659'],
        # force_test_shots=['145388', '145422'],
        predict_shots=['145384', '145385', '145391', '145410', '145420', '145422', '145427', '157303', '157322', '157372', '157374', '158076', '189191', '189189', '189199', '203659', '203663', '145388', '145419', '157376', '200635', '203671', '145387', '145425', '157323', '157373', '157375', '157377', '159443', '200021', '203660', '203672', '203665', '203667'],
    )

    weight_decay = 0.001
    trial_name = f'{logger_hash}'

    lightning_model = elm_lightning_model.Lightning_Model(
        encoder_lr=1e-3,
        decoder_lr=1e-4,
        signal_window_size=datamodule.signal_window_size,
        n_rows=datamodule.n_rows,
        n_cols=datamodule.n_cols,
        monitor_metric='sum_loss/val',
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=10,
        weight_decay=weight_decay,
        encoder_type='none',
        cnn_nlayers=2,
        cnn_num_kernels=[16, 32],
        cnn_kernel_time_size=[8, 4],
        cnn_kernel_spatial_size=[3, 3],
        cnn_padding = [1, 1],
        cnn_maxpool_spatial_size = [2, 1],
        cnn_maxpool_time_size = [2, 2],
        fft_num_kernels=10,
        fft_nbins=2,
        fft_subwindows=2,
        fft_kernel_freq_size=5,
        fft_kernel_spatial_size=[(3, 3) for _ in range(1)],
        fft_maxpool_freq_size=4,
        fft_maxpool_spatial_size=[(1, 2) for _ in range(1)],
        fft_dropout=0.1,
        leaky_relu_slope=0.001,
        mlp_layers=(50, 50),
        mlp_dropout=0.1,
        multiclass_classifier_mlp=True,
        num_classes=datamodule.num_classes,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
        # num_val_batches=1,
        # val_check_interval=5000,
        num_train_batches=2499,
    )

    trainer.run_all(
        max_epochs=60,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=100,
        skip_test=False,
        skip_predict=False,
        float_precision=32,
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