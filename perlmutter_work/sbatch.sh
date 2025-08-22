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
#SBATCH --qos=debug

###SBATCH --signal=SIGTERM@1200

#SBATCH --array=1-2

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
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    feature_model_layers = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        # {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        # {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
    )
    mlp_tasks={
        'elm_class': [None,32,1],
        'conf_onehot': [None,32,4],
    }
    main(
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        experiment_name='experiment_v8',
        feature_model_layers=feature_model_layers,
        mlp_tasks=mlp_tasks,
        max_elms=80,
        max_epochs=20,
        lr_warmup_epochs=4,
        fraction_validation=0.2,
        num_workers=8,
        max_confinement_event_length=int(15e3),
        confinement_dataset_factor=0.2,
        monitor_metric='sum_loss/train',
        fir_bp=(None, 200),
        use_wandb=True,
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
