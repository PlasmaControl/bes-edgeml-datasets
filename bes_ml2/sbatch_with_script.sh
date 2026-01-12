#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=8
#SBATCH --time=07:30:00
#SBATCH --qos=regular
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

    n_rows = 8
    n_cols = 8
    num_classes = 4

    datamodule = confinement_datamodule_4.Confinement_Datamodule(
            data_file = '/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20240112.hdf5',
            # max_shots_per_class=35,
            num_classes = num_classes,
            signal_window_size=12,
            n_rows = n_rows,
            n_cols = n_cols,
            fraction_test=0.15,
            fraction_validation=0.125,
            batch_size=256,
            plot_data_stats=False,
            num_workers=4,
            seed=0,
            world_size=world_size,
            # lower_cutoff_frequency_hz=2.5e3,
            # upper_cutoff_frequency_hz=150e3,
            one_hot_labels=True,
            bad_shots=['169865', '169869', '170012', '170015', '170021', '174656', '174658', '174660', '175514', '184427', '184822', '192707',
                '152814', '179205', '179209', '179211', '179212', '179213', '179214', '179216', '179219', '179220', '179223', '179314', '179321', '179328', '179331', '179333', '179334', '180354', '180363', '186230'],
            force_test_shots=['149996', '164884', '163505'],
            # force_validation_shots=['184443', '191783', '196037', '185474', '164884', '175555', '159182', '189301',
            #     '157092', '184773', '159175', '171474', '159553', '164880', '185491', '171472',
            #     '175833', '185502', '184467', '192710', '174618', '175826', '159544', '179712',
            #     '184449', '165113', '184818', '184968', '184813', '182627', '185918', '159161',
            #     '160086', '195626', '182682', '174619', '159605', '162905', '195642', '169371',
            #     '175567', '171475', '169877', '184810'],
            # force_test_shots=['175490', '187035', '191376', '164869', '171471', '172212', '184463', '160778',
            #     '184451', '192942', '175591', '174675', '169848','169859', '187045', '175564',
            #     '160777', '184467', '185498', '192761', '184454', '184441', '165113', '192767',
            #     '189189', '175599', '163469', '171472', '189196', '169350', '195642', '195550',
            #     '169369', '184453', '161320', '160060', '189106', '163509', '163460', '169387',
            #     '195825', '184813', '159536', '149996', '164884', '163505'],
        )

    weight_decay = 1e-4
    cnn_nlayers = 2
    fft_nlayers = 4
    trial_name = f'{logger_hash}'

    lightning_model = elm_lightning_model.Lightning_Model(
        encoder_lr=1e-4,
        decoder_lr=1e-6,
        signal_window_size=datamodule.signal_window_size,
        n_rows=datamodule.n_rows,
        n_cols=datamodule.n_cols,
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=10,
        weight_decay=weight_decay,
        cnn_dropout=0.1,
        encoder_type='none',
        cnn_num_kernels=(10, 8),
        cnn_nlayers=cnn_nlayers,
        cnn_kernel_spatial_size=3,
        cnn_kernel_time_size=[2] * cnn_nlayers,
        cnn_maxpool_time_size=(4, 2),
        cnn_maxpool_spatial_size=[(1, 1), (1, 1)],
        fft_nlayers=fft_nlayers,
        # use_phase=True,
        fft_num_kernels=10,
        fft_subwindows=1,
        fft_nbins=1,
        # fft_kernel_freq_size=5,
        # fft_kernel_spatial_size=[(3, 3) for _ in range(fft_nlayers)],
        # fft_maxpool_freq_size=4,
        # fft_maxpool_spatial_size=[(1, 2) for _ in range(fft_nlayers)],
        # fft_maxpool_spatial_size=[(1, 2), (2, 1), (1, 1)],
        # fft_padding="same",
        fft_kernel_freq_size=3,
        fft_kernel_spatial_size=3,
        fft_dropout=0.1,
        leaky_relu_slope=0.001,
        mlp_layers=(50, 50),
        # mlp_dropout=0.1,
        # temperature=0.75,
        multiclass_classifier_mlp=True,
        reconstruction_decoder=False,
        time_to_elm_mlp=False,
        classifier_mlp=False,
        velocimetry_mlp=False,
        # visualize_embeddings=True,
        num_classes = num_classes,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
        # num_train_batches=1,
        # num_val_batches=1,
        num_train_batches=2000, 
        val_check_interval=1000,
    )

    trainer.run_all(
        max_epochs=120,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=40,
        # skip_test=True,
        # skip_predict=True,
        # debug_predict=True,
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