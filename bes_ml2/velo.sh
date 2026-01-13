#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=david.smith@wisc.edu
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

JOB_DIR=${HOME}/scratch-ml
mkdir --parents $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

export WANDB__SERVICE_WAIT=300

PYTHON_SCRIPT=$(cat << END

import sys
import os
import time

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
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

    block_cols = [1, 3, 5, 7]
    row_stride   = 1
    row_offset   = 4
    R_sel = len(np.arange(8)[row_offset::row_stride])
    C_sel = len(block_cols)

    good_times_psi_93 = {
        # 145384: [()], # this means use all times
        # 145387: [()],
        # 145388: [()],
        145391: [(1900, 4200)],
        # 145410: [()],
        # 145419: [()],
        # 145420: [()],
        # 145422: [()],
        # 145425: [()],
        # 145427: [()],
        # 157303: [()],
        # 157322: [()],
        # 157323: [()],
        # 157372: [()],
        # 157373: [()],
        # 157374: [()],
        157375: [(1900, 3600), (4000, 5500)],
        # 157376: [()],
        157377: [(1800, 5000), (5400, 6000)],
        # 158076: [()],
        159443: [(1900, 5600)],
        189189: [(2000, 4700)],
        # 189191: [()],
        189199: [(1800, 3000), (4200, 4700)],
        # 200021: [()],
        # 200632: [()],
        # 200634: [()],
        200637: [(1100, 3800)],
        200638: [(800, 3500)],
        200639: [(800, 4600)],
        200643: [(800, 4600)],
        # 203152: [()],
        # 203416: [()],
        203417: [(2250, 2700), (2900, 3300), (3600, 4100)],
        # 203418: [()],
        203419: [(2250, 3500), (3750, 4000)],
        # 203420: [()],
        203423: [(2300, 3850)],
        203469: [(600, 4600)],
        203470: [(500, 3900)],
        203471: [(500, 3800)],
        # 203475: [()],
        203483: [(800, 4300)],
        203484: [(800, 4200)],
        203485: [(800, 2100), (3200, 4500)],
        # 203659: [()],
        203660: [(1100, 2900)],
        # 203662: [()],
        203663: [(1100, 3900)],
        203664: [(1000, 4000)],
        # 203665: [()],
        # 203667: [()],
        # 203671: [()],
        # 203672: [()],
        203946: [(4000, 5400)],
        204286: [(1800, 4600)],
        204287: [(1500, 4000)],
        # 204288: [()],
        204289: [(1100, 4300)],
        204290: [(1100, 4100)],
        204291: [(1100, 3700)],
        # 204292: [()],
        # 204293: [()],
        204294: [(1100, 4100)],
        # 204295: [()],
        # 204296: [()],
        # 204297: [()],
        204299: [(1500, 4500)],
        # 204301: [()],
        204302: [(1800, 4100)],
        204303: [(1600, 4400)],
        # 204837: [()],
        205867: [(2200, 4400)],

    }

    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
            data_file='/global/cfs/cdirs/m3586/kgill/velocimetry_data/20251027_raw_signals_psi_interp.hdf5',
            signal_window_size=48,
            batch_size=1024,
            num_workers=4,
            seed=0,
            world_size=world_size,
            standardize_signals=False,
            split_method='shot',
            train_shots = [
                # 145xxx
                '145384', '145420', '145425',
                # 157xxx 
                '157303', '157372', '157375', '158076',
                # 200xxx
                '200643',
                # Early 203xxx 
                '203416', '203420', '203483',
                # Late 203xxx + 204xxx
                '203664', '203671', '204292', '204837',
            ],
            validation_shots = [
                '145388', '145419',      # 145xxx
                '157322', '157373',      # 157xxx  
                '203417', '203423',      # early 203xxx
                '203665', '203672',      # late 203xxx
            ],            
            test_shots = ['145387', '145391', '145410', '145422', '145427', '157323', '157374', '157377', '159443', '189189', '189191', '189199', '200634', '200637', '200638', '200639', '203152', '203418', '203419', '203469', '203470', '203471', '203484', '203659', '203660', '203663', '203667', '203946', '204286', '204287', '204288', '204289', '204290', '204291', '204294',  '204296', '204297', '204299', '204301', '204302', '204303'],
            predict_shots = ['145384', '145420', '145425', '157303', '157372', '157375', '158076', '200643', '203416', '203420', '203483', '203664', '203671', '204292', '204295', '204837', '145388', '145419', '157322', '157373', '157376', '203417', '203423', '203475', '203485', '203665', '203672', '204293', '145387', '145391', '145410', '145422', '145427', '157323', '157374', '157377', '159443', '189189', '189191', '189199', '200634', '200637', '200638', '200639', '203152', '203418', '203419', '203469', '203470', '203471', '203484', '203659', '203660', '203663', '203667', '203946', '204286', '204287', '204288', '204289', '204290', '204291', '204294',  '204296', '204297', '204299', '204301', '204302', '204303'],
            split_train_data_per_gpu=True,
            do_flip_augmentation=True, # crucial
            shot_time_windows = good_times_psi_93, # good times
            # --- block + label knobs ---
            block_cols=block_cols,
            row_stride=row_stride,
            row_offset=row_offset,
            target_sampling_hz=1_000_000.0,
            label_target_psi=0.93, # 0.85, 0.88, 0.91, 0.93, 0.95
            label_tolerance_ms=0.6,
            window_hop=1,
            n_rows=R_sel,   
            n_cols=C_sel,   
    )

    weight_decay = 0.00001
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
        mlp_dropout=0.001,
        velocimetry_mlp=True,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
    )

    trainer.run_all(
        max_epochs=3,
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
srun --unbuffered python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE