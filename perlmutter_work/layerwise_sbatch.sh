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

#SBATCH --signal=SIGTERM@300

#SBATCH --array=0-19%4

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

# export rand=${RANDOM}
# echo Random number: ${rand}

SCRIPT=$(cat << END
import os
from model_trainer.main_multitask import main

if __name__=='__main__':
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    jobs = (
        ('r41506284_1_2025_08_08_14_45_37', (None, 250)),
        ('r41506275_1_2025_08_08_14_18_51', (None, 250)),
        ('r41506362_1_2025_08_08_18_06_08', (None, 150)),
        ('r41506372_1_2025_08_08_18_23_25', (None, 150)),
        ('r41506318_1_2025_08_08_15_59_47', (None, 150)),
        ('r41506344_1_2025_08_08_16_49_14', (None, 150)),
        ('r41506354_1_2025_08_08_17_32_48', (None, 150)),
        ('r41506295_1_2025_08_08_15_31_50', (None, 150)),
        ('r41506326_1_2025_08_08_16_49_15', (None, 150)),
        ('r41506381_1_2025_08_08_18_23_18', (None, 150)),
        ('r41506423_1_2025_08_08_18_43_50', (8, 150)),
        ('r41506431_1_2025_08_08_18_49_38', (8, 150)),
        ('r41506405_1_2025_08_08_18_37_17', (8, 150)),
        ('r41506394_1_2025_08_08_18_31_08', (8, 150)),
        ('r41506588_1_2025_08_08_19_08_19', (8, None)),
        ('r41506533_1_2025_08_08_18_55_49', (8, None)),
        ('r41506580_1_2025_08_08_19_08_38', (8, None)),
        ('r41506564_1_2025_08_08_19_04_42', (8, None)),
        ('r41506544_1_2025_08_08_18_59_42', (8, None)),
        ('r41506554_1_2025_08_08_19_03_23', (8, None)),
    )
    job_id = jobs[i_array][0]
    fir_bp = jobs[i_array][1]
    main(
        # scenario
        signal_window_size=256,
        experiment_name='experiment_256_v7',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        max_elms=40,
        # model
        use_optimizer='adam',
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks = {
            'elm_class': [None, 32, 1],
        },
        no_bias=False,
        monitor_metric='elm_class/bce_loss/train',
        batch_norm=True,
        # training
        max_epochs=1000,
        lr=3e-2,
        lr_warmup_epochs=15,
        lr_scheduler_patience=80,
        deepest_layer_lr_factor=1./2,
        batch_size={0:128, 15:256, 120:512, 400:1024},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=250,
        transfer_model=f'experiment_256_v6/{job_id}/checkpoints/last.ckpt',
        fir_bp=fir_bp,
        transfer_max_layer=10,
        transfer_layer_lr_factor=1./10,
    )
END
)

echo Script:
echo "${SCRIPT}"


# SCRIPT=/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/layerwise_learning.py
# echo Script: ${SCRIPT}
# cat ${SCRIPT}

JOB_DIR=${HOME}/scratch-ml
echo Job directory: ${JOB_DIR}

cd $JOB_DIR || exit
echo Current directory: ${PWD}

START_TIME=$(date +%s)
srun --unbuffered python -c "${SCRIPT}"
# srun --unbuffered python ${SCRIPT}
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min

echo Exit code: ${EXIT_CODE}
