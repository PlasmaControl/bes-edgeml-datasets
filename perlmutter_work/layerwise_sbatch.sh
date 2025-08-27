#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=david.smith@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --time=60
#SBATCH --qos=regular

#SBATCH --signal=SIGTERM@200

#SBATCH --array=1-10%5

module --redirect list
which python

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
echo UNIQUE_IDENTIFIER: ${UNIQUE_IDENTIFIER}

export WANDB__SERVICE_WAIT=500

export rand=${RANDOM}
echo Random number: ${rand}

SCRIPT=$(cat << END
import os
from numpy import random
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    jobs = (
        ('r42002857_14_2025_08_25_11_30_57', (None, 200)),
        ('r42002857_45_2025_08_25_16_44_36', (None, 200)),
        ('r42010042_43_2025_08_25_23_36_33', (None, 200)),
        ('r42010042_13_2025_08_25_19_52_47', (8, None)),
        ('r42002857_11_2025_08_25_10_52_35', (8, None)),
        ('r42002857_28_2025_08_25_13_44_37', (8, None)),
        ('r42010042_7_2025_08_25_19_09_05', (8, None)),
        ('r42010042_22_2025_08_25_20_56_00', (8, 200)),
        ('r42010042_11_2025_08_25_19_23_42', (8, 200)),
        ('r42010042_41_2025_08_25_23_22_31', (8, 200)),
    )
    job_id = jobs[i_array][0]
    fir_bp = jobs[i_array][1]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v12',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=40,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.3,
        fraction_validation=0.15,
        # model
        use_optimizer='adam',
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None,32,1],
            'conf_onehot': [None,32,4],
        },
        monitor_metric='sum_loss/train',
        fir_bp=fir_bp,
        # training
        max_epochs=300,
        log_freq=100,
        lr=1e-2,
        lr_warmup_epochs=20,
        deepest_layer_lr_factor=0.1,
        lr_scheduler_patience=90,
        weight_decay=1e-4,
        batch_size={0:128, 20:256, 70:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
        backbone_model_path=f'multi_256_v10/{job_id}',
        backbone_first_n_layers = 3,
        backbone_initial_ratio_lr = 0.01,
        backbone_unfreeze_at_epoch = 50,
        backbone_warmup_rate = 2,
    )
END
)

echo Script:
echo "${SCRIPT}"


JOB_DIR=${HOME}/scratch-ml
echo Job directory: ${JOB_DIR}

cd $JOB_DIR || exit
echo Current directory: ${PWD}

START_TIME=$(date +%s)
srun --unbuffered python -c "${SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min

echo Exit code: ${EXIT_CODE}
