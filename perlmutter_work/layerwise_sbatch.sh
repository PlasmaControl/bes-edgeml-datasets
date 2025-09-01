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

#SBATCH --array=1-80%8

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
    seed = os.environ.get('rand', None)
    if seed is not None: seed = int(seed)
    print(f'RNG seed: {seed}')
    rng = random.default_rng(seed=seed)

    fir_choices = (
        (8, None),
        (None, 200),
        (8, 200),
    )

    # mlp_choices = (
    #     [None,],
    #     [None, 16],
    #     [None, 32],
    # )
    # mlp_layers = mlp_choices[rng.choice(len(mlp_choices))]
    mlp_layers = [None, 32]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v15',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        # max_elms=rng.choice([40, 60]),
        max_elms=80,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.3,
        fraction_validation=0.15,
        # model
        use_optimizer='adam',
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        ),
        mlp_tasks={
            'elm_class': mlp_layers+[1,],
            'conf_onehot': mlp_layers+[4,],
        },
        monitor_metric='elm_class/bce_loss/train',
        fir_bp=fir_choices[rng.choice(len(fir_choices))],
        # training
        max_epochs=500,
        log_freq=100,
        lr=1e-2,
        lr_warmup_epochs=20,
        lr_scheduler_patience=100,
        deepest_layer_lr_factor=1.,
        weight_decay=1e-5,
        batch_size={0:128, 20:256, 80:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
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
