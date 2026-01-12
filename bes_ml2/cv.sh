#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --job-name=cv_velo
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-4

# Cross-validation for BES velocimetry model
# Groups shots by similar characteristics (nearby shot numbers)
# Uses 5-fold CV where each fold holds out one group for validation

export UNIQUE_IDENTIFIER="${SLURM_ARRAY_JOB_ID}_fold${SLURM_ARRAY_TASK_ID}"
export CV_FOLD=${SLURM_ARRAY_TASK_ID}

echo "=== Cross-Validation Fold ${CV_FOLD} ==="
echo "Unique ID: ${UNIQUE_IDENTIFIER}"

# Run with srun - use quoted heredoc to prevent variable expansion
srun --ntasks=8 --gpus=8 python3 - << 'PYTHON_SCRIPT'
import sys
import os
import time

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
from bes_ml2 import elm_lightning_model

logger_hash = os.getenv('UNIQUE_IDENTIFIER', '0')
world_size = int(os.getenv('SLURM_NTASKS', 0))
world_rank = int(os.getenv('SLURM_PROCID', 0))
local_rank = int(os.getenv('SLURM_LOCALID', 0))
node_rank = int(os.getenv('SLURM_NODEID', 0))
fold = int(os.getenv('CV_FOLD', 0))
print(f'Fold {fold}: World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})')

is_global_zero = world_rank == 0

if not is_global_zero:
    f = open(os.devnull, 'w')
    sys.stdout = f

try:
    t_start = time.time()

    # --- Your selection knobs ---
    block_cols = [1, 3, 5, 7]
    row_stride = 1
    row_offset = 4

    R_sel = len(np.arange(8)[row_offset::row_stride])
    C_sel = len(block_cols)

    good_times_psi_93 = {
        145391: [(1900, 4200)],
        157375: [(1900, 3600), (4000, 5500)],
        157377: [(1800, 5000), (5400, 6000)],
        159443: [(1900, 5600)],
        189189: [(2000, 4700)],
        189199: [(1800, 3000), (4200, 4700)],
        200637: [(1100, 3800)],
        200638: [(800, 3500)],
        200639: [(800, 4600)],
        200643: [(800, 4600)],
        203417: [(2250, 2700), (2900, 3300), (3600, 4100)],
        203419: [(2250, 3500), (3750, 4000)],
        203423: [(2300, 3850)],
        203469: [(600, 4600)],
        203470: [(500, 3900)],
        203471: [(500, 3800)],
        203483: [(800, 4300)],
        203484: [(800, 4200)],
        203485: [(800, 2100), (3200, 4500)],
        203660: [(1100, 2900)],
        203663: [(1100, 3900)],
        203664: [(1000, 4000)],
        203946: [(4000, 5400)],
        204286: [(1800, 4600)],
        204287: [(1500, 4000)],
        204289: [(1100, 4300)],
        204290: [(1100, 4100)],
        204291: [(1100, 3700)],
        204294: [(1100, 4100)],
        204299: [(1500, 4500)],
        204302: [(1800, 4100)],
        204303: [(1600, 4400)],
    }

    # Define shot groups based on shot number ranges (similar characteristics)
    shot_groups = {
        0: ['145384', '145388', '145419', '145420', '145425'],  # 145xxx series
        1: ['157303', '157322', '157372', '157373', '157375', '157376', '158076'],  # 157xxx series
        2: ['200643'],  # 200xxx series
        3: ['203416', '203417', '203420', '203423', '203475', '203483', '203485'],  # 203xxx early
        4: ['203664', '203665', '203671', '203672', '204292', '204293', '204295', '204837'],  # 203xxx late + 204xxx
    }

    # Test shots remain fixed
    test_shots = [
        '145387', '145391', '145410', '145422', '145427', '157323', '157374', '157377',
        '159443', '189189', '189191', '189199', '200634', '200637', '200638', '200639',
        '203152', '203418', '203419', '203469', '203470', '203471', '203484', '203659',
        '203660', '203663', '203667', '203946', '204286', '204287', '204288', '204289',
        '204290', '204291', '204294', '204296', '204297', '204299', '204301', '204302', '204303'
    ]

    # Build train and validation sets based on fold
    # The fold number determines which group is held out for validation
    validation_shots = shot_groups[fold]
    train_shots = []
    for group_id, shots in shot_groups.items():
        if group_id != fold:
            train_shots.extend(shots)

    # Combine all for predict
    predict_shots = train_shots + validation_shots + test_shots

    print(f"Fold {fold}: Training on {len(train_shots)} shots, validating on {len(validation_shots)} shots")
    print(f"  Train shots: {train_shots}")
    print(f"  Val shots: {validation_shots}")

    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
        data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20251027_raw_signals_psi_interp.hdf5',
        signal_window_size=48,
        batch_size=256,
        num_workers=4,
        seed=0,
        world_size=world_size,
        standardize_signals=False,
        split_method='shot',
        train_shots=train_shots,
        validation_shots=validation_shots,
        test_shots=test_shots,
        predict_shots=predict_shots,
        split_train_data_per_gpu=True,
        vZ_uncertainty_threshold=45.0,
        target_labels=["vZ", "vZ_uncertainty"],
        do_flip_augmentation=True,
        shot_time_windows=good_times_psi_93,
        block_cols=block_cols,
        row_stride=row_stride,
        row_offset=row_offset,
        target_sampling_hz=1_000_000.0,
        label_target_psi=0.93,
        label_tolerance_ms=0.6,
        window_hop=1,
        n_rows=R_sel,
        n_cols=C_sel,
    )

    weight_decay = 0.001
    trial_name = f'{logger_hash}'

    lightning_model = elm_lightning_model.Lightning_Model(
        encoder_lr=1e-3,
        decoder_lr=1e-3,
        signal_window_size=datamodule.signal_window_size,
        n_rows=R_sel,
        n_cols=C_sel,
        monitor_metric='sum_loss/val',
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=5,
        weight_decay=weight_decay,
        encoder_type='none',
        mlp_layers=(50, 50),
        mlp_dropout=0.01,
        velocimetry_mlp=True,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_cv',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
        num_train_batches=49500,
    )

    trainer.run_all(
        max_epochs=600,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=100,
        skip_test=False,
        skip_predict=False,
        float_precision=32,
    )
    print(f'Fold {fold}: Python elapsed time {(time.time()-t_start)/60:.1f} min')
except:
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
    raise
finally:
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
PYTHON_SCRIPT
