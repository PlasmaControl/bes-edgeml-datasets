#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=david.smith@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --signal=SIGTERM@180

#SBATCH --time=0-4
#SBATCH --qos=regular
#SBATCH --array=0-12%8

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

SCRIPT=$(cat << END
import os
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    jobs = (
        ('r43166502_0_2025_09_26_20_00_19', (None, 100), 300, 1e-4),
        ('r43166502_1_2025_09_26_20_00_50', (None, 100), 300, 1e-4),
        ('r43166502_2_2025_09_26_20_12_44', (12, None), 300, 1e-5),
        ('r43166502_3_2025_09_26_20_12_42', (None, 100), 300, 1e-4),
        ('r43166502_4_2025_09_26_20_33_30', (12, None), 300, 1e-5),
        ('r43166502_5_2025_09_26_20_33_35', (None, 100), 300, 1e-4),
        ('r43166502_6_2025_09_26_20_47_37', (None, 100), 300, 1e-5),
        ('r43166502_7_2025_09_26_20_47_33', (None, 200), 400, 1e-5),
        ('r43166502_8_2025_09_26_21_30_57', (None, 200), 400, 1e-5),
        ('r43166502_9_2025_09_26_21_32_15', (None, 200), 400, 1e-5),
        ('r43166502_10_2025_09_26_22_50_13', (None, 200), 400, 1e-5),
        ('r43166502_11_2025_09_26_22_54_55', (None, 200), 400, 1e-5),
        ('r43166502_12_2025_09_26_23_09_22', (None, 200), 400, 1e-5),
    )
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    i_job = i_array % len(jobs)
    job_id = jobs[i_job][0]
    fir_bp = jobs[i_job][1]
    max_elms = jobs[i_job][2]
    weight_decay = jobs[i_job][3]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v24',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=max_elms,
        max_confinement_event_length=int(50e3),
        confinement_dataset_factor=0.6,
        fraction_validation=0.15,
        num_workers=4,
        # model
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None, 32, 1],
            'conf_onehot': [None, 32, 4],
        },
        monitor_metric='sum_score/train',
        fir_bp=fir_bp,
        # training
        max_epochs=800,
        lr=3e-3,
        lr_warmup_epochs=20,
        lr_scheduler_patience=100,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=512,
        use_wandb=True,
        early_stopping_patience=250,
        backbone_model_path=f'multi_256_v23/{job_id}',
        backbone_first_n_layers=100,
        backbone_unfreeze_at_epoch=0,
        backbone_initial_lr=3e-5,
        backbone_warmup_rate=2,
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
