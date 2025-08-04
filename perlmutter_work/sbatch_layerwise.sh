#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=david.smith@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --time=45
#SBATCH --qos=regular

#SBATCH --signal=SIGTERM@300

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

rand=${RANDOM}
echo Random number: ${rand}

# SCRIPT=$(cat << END
# from numpy import random
# from model_trainer.main_multitask import main

# if __name__=='__main__':
#     print(f'Linux rand: { ${rand} }')
#     rng = random.default_rng(seed=${rand})

#     kwargs = {}
#     kwargs['feature_model_layers'] = (
#         {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
#         {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
#         {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
#     )

#     main(
#         # scenario
#         signal_window_size=256,
#         experiment_name='experiment_256_v5',
#         # data
#         elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
#         max_elms=rng.choice([40, 60, 80]),
#         # model
#         use_optimizer='adam',
#         mlp_tasks = {
#             'elm_class': [None, 32, 1],
#         },
#         no_bias=rng.choice([True, False]),
#         fir_bp_low=rng.choice([0, 10]),
#         fir_bp_high=rng.choice([0, 75, 250]),
#         monitor_metric='elm_class/bce_loss/train',
#         batch_norm=True,
#         # training
#         max_epochs=500,
#         log_freq=100,
#         lr=rng.choice([1e-2, 3e-2]),
#         # lr=3e-2,
#         lr_warmup_epochs=15,
#         lr_scheduler_patience=80,
#         deepest_layer_lr_factor=1.,
#         batch_size={0:64, 15:128, 30:256},
#         num_workers=8,
#         gradient_clip_val=1,
#         gradient_clip_algorithm='value',
#         use_wandb=True,
#         early_stopping_patience=200,
#         # kwargs
#         **kwargs,
#     )
# END
# )

# echo Script:
# echo "${SCRIPT}"


SCRIPT=/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/layerwise_learning.py
echo Script: ${SCRIPT}
cat ${SCRIPT}

JOB_DIR=${HOME}/scratch-ml
echo Job directory: ${JOB_DIR}

cd $JOB_DIR || exit
echo Current directory: ${PWD}

START_TIME=$(date +%s)
# srun --unbuffered python -c "${SCRIPT}"
srun --unbuffered python ${SCRIPT}
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min

echo Exit code: ${EXIT_CODE}
