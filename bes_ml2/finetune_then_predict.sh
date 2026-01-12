#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --qos=debug
###SBATCH --array=0

# -------------------------
# USER KNOBS 
# -------------------------
PRETRAIN_CKPT="/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/47160693/checkpoints/epoch=12-step=581815.ckpt"
FT_MAX_EPOCHS=50
FT_ENCODER_LR="1e-3"
FT_DECODER_LR="1e-3"
# -------------------------

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
from pathlib import Path

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
from bes_ml2 import elm_lightning_model

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
world_size = int(os.getenv('SLURM_NTASKS', 1))
world_rank = int(os.getenv('SLURM_PROCID', 0))
local_rank = int(os.getenv('SLURM_LOCALID', 0))
node_rank = int(os.getenv('SLURM_NODEID', 0))
print(f'World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})')

is_global_zero = (world_rank == 0)

# Quiet non-rank0 stdout (keeps logs readable)
if not is_global_zero:
    f = open(os.devnull, 'w')
    sys.stdout = f

def _get_int(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            try:
                return int(getattr(obj, n))
            except Exception:
                pass
    return default

try:
    t_start = time.time()

    # -------------------------
    # 0) Point to pretrained ckpt
    # -------------------------
    pretrain_ckpt = os.getenv("PRETRAIN_CKPT", "").strip()
    if not pretrain_ckpt:
        raise RuntimeError("PRETRAIN_CKPT env var is not set (path to .ckpt).")

    pretrain_ckpt = str(Path(pretrain_ckpt).expanduser().resolve())
    if not Path(pretrain_ckpt).exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_ckpt}")

    # -------------------------
    # 1) Load pretrained model (weights + hparams)
    # -------------------------
    lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(
        checkpoint_path=pretrain_ckpt,
        map_location="cpu",   # Lightning will move to GPU
        strict=True,
    )

    # OPTIONAL: override fine-tune LR without changing architecture
    FT_ENCODER_LR = float(os.getenv("FT_ENCODER_LR", "1e-4"))
    FT_DECODER_LR = float(os.getenv("FT_DECODER_LR", "1e-4"))
    if hasattr(lightning_model, "hparams"):
        try:
            lightning_model.hparams.encoder_lr = FT_ENCODER_LR
            lightning_model.hparams.decoder_lr = FT_DECODER_LR
        except Exception:
            pass
    if hasattr(lightning_model, "encoder_lr"): lightning_model.encoder_lr = FT_ENCODER_LR
    if hasattr(lightning_model, "decoder_lr"): lightning_model.decoder_lr = FT_DECODER_LR

    # Pull key shape knobs from checkpoint so you cannot accidentally mismatch
    W_model   = _get_int(lightning_model.hparams, ["signal_window_size"], default=_get_int(lightning_model, ["signal_window_size"], 48))
    n_rows_m  = _get_int(lightning_model.hparams, ["n_rows"], default=_get_int(lightning_model, ["n_rows"], None))
    n_cols_m  = _get_int(lightning_model.hparams, ["n_cols"], default=_get_int(lightning_model, ["n_cols"], None))

    # -------------------------
    # 2) block selection knobs (MUST match what the pretrained model expects)
    # -------------------------
    block_cols  = [1, 3, 5, 7]
    row_stride  = 1
    row_offset  = 4

    R_sel = len(np.arange(8)[row_offset::row_stride])
    C_sel = len(block_cols)

    if (n_rows_m is not None) and (R_sel != n_rows_m):
        raise RuntimeError(f"n_rows mismatch: checkpoint has {n_rows_m}, but your selection gives {R_sel}")
    if (n_cols_m is not None) and (C_sel != n_cols_m):
        raise RuntimeError(f"n_cols mismatch: checkpoint has {n_cols_m}, but your selection gives {C_sel}")
    if W_model != int(W_model):
        raise RuntimeError("Bad signal_window_size from checkpoint?")

    # -------------------------
    # 3) Choose fine-tune + predict shots 
    # -------------------------
    FT_TRAIN_SHOTS = ["205869"]
    FT_VAL_SHOTS = ["205867"]
    FT_TEST_SHOTS = ["205870"]
    PREDICT_SHOTS = ["205867", "205869", "205870"]

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

    # -------------------------
    # 5) Build datamodule for fine-tune + predict
    # -------------------------
    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
        data_file="/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/205867_psi_interp.hdf5",
        signal_window_size=W_model,
        batch_size=1024,
        num_workers=4,
        seed=0,
        world_size=world_size,
        standardize_signals=False,
        split_method="shot",
        train_shots=FT_TRAIN_SHOTS,
        validation_shots=FT_VAL_SHOTS,
        test_shots=FT_TEST_SHOTS,
        predict_shots=PREDICT_SHOTS,
        split_train_data_per_gpu=False,   
        do_flip_augmentation=True,
        shot_time_windows=good_times_psi_93,
        block_cols=block_cols,
        row_stride=row_stride,
        row_offset=row_offset,
        target_sampling_hz=1_000_000.0,
        label_target_psi=0.68,
        label_tolerance_ms=0.6,
        window_hop=1,
        predict_window_stride=48,
        n_rows=R_sel,
        n_cols=C_sel,
    )

    # -------------------------
    # 6) Trainer wrapper (reuse BES_Trainer)
    # -------------------------
    trial_name = f"ft_{logger_hash}"

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir="./exp_gill01",
        trial_name=trial_name,
        wandb_log=True,   
        log_freq=100,
    )

    # Fine-tune settings
    FT_MAX_EPOCHS = int(os.getenv("FT_MAX_EPOCHS", "10"))

    trainer.run_all(
        max_epochs=FT_MAX_EPOCHS,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=50,
        skip_test=False,         # set False if you want test after fine-tune
        skip_predict=False,     # we DO want predictions
        float_precision=32,
    )

    if is_global_zero:
        print(f"Loaded ckpt: {pretrain_ckpt}")
        print(f"Fine-tuned best ckpt: {trainer.best_model_path}")
        print(f"Fine-tuned last ckpt: {trainer.last_model_path}")
        print(f"Python elapsed time {(time.time()-t_start)/60:.1f} min")

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
srun --export=ALL,\
PRETRAIN_CKPT="$PRETRAIN_CKPT",\
FT_MAX_EPOCHS="$FT_MAX_EPOCHS",\
FT_ENCODER_LR="$FT_ENCODER_LR",\
FT_DECODER_LR="$FT_DECODER_LR" \
python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE
