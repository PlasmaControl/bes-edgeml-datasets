#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=david.smith@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --time=30
#SBATCH --qos=regular

#SBATCH --signal=SIGTERM@200

#SBATCH --array=1-50%5

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

rand=${RANDOM}
echo Random number: ${rand}

SCRIPT=$(cat << END
from numpy import random
from model_trainer.main_multitask import main

if __name__=='__main__':
    print(f'Linux rand: { ${rand} }')
    rng = random.default_rng(seed=${rand})

    kwargs = {}
    kwargs['feature_model_layers'] = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
    )
    mlp_hidden_layer_size = rng.choice([16, 32])
    kwargs['mlp_tasks'] = {'elm_class': [None, mlp_hidden_layer_size, 1]}

    regularization_choice = rng.choice(3)
    if regularization_choice == 0:
        kwargs['batch_norm'] = True
    elif regularization_choice == 1:
        kwargs['dropout'] = rng.choice([0.04, 0.12])

    main(
        # scenario
        signal_window_size=256,
        experiment_name='experiment_256_v1',
        # restart
        restart_trial_name='',
        wandb_id='',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        max_elms=rng.choice([20, 40, 80]),
        # model,
        no_bias=rng.choice([True, False]),
        # fir_bp_low=5,
        fir_bp_low=rng.choice([0, 10]),
        # fir_bp_high=250,
        fir_bp_high=rng.choice([0, 75, 250]),
        # training
        max_epochs=500,
        log_freq=100,
        # lr=1e-4,
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
        lr_warmup_epochs=5,
        lr_scheduler_patience=50,
        # deepest_layer_lr_factor=0.5,
        deepest_layer_lr_factor=rng.choice([1.0, 0.2]),
        batch_size={0:64, 10:128, 20:256},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
        # kwargs
        **kwargs,
    )
END
)

echo Script:
echo "${SCRIPT}"


# SCRIPT=${PWD}/scenarios_sw128.py
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
