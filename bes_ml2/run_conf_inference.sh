#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=2
#SBATCH --array=0-4
#SBATCH --time=00:30:00
#SBATCH --qos=debug

# ---------------------- USER CONTROLS ----------------------
# You can pass 2 positional args to this script:
#   $1 -> N_SHOTS_PER_JOB (default 2)
#   $2 -> MODEL_ID        (default 41825238)
# Or set env vars N_SHOTS_PER_JOB / MODEL_ID when calling sbatch.

N_SHOTS_PER_JOB="${1:-${N_SHOTS_PER_JOB:-2}}"
MODEL_ID="${2:-${MODEL_ID:-41825238}}"

# Optional: if you know the ckpt filename, set CKPT_NAME, e.g.
# CKPT_NAME="epoch=2-step=42873.ckpt"
# Otherwise the script will pick the most recently modified .ckpt.
CKPT_NAME="${CKPT_NAME:-}"

# ------------------ STANDARD DIAGNOSTICS -------------------
echo Python executable: "$(which python)"
echo
echo Job name: "$SLURM_JOB_NAME"
echo QOS: "$SLURM_JOB_QOS"
echo Account: "$SLURM_JOB_ACCOUNT"
echo Submit dir: "$SLURM_SUBMIT_DIR"
echo
echo Job array ID: "$SLURM_ARRAY_JOB_ID"
echo Job ID: "$SLURM_JOBID"
echo Job array task: "$SLURM_ARRAY_TASK_ID"
echo Job array task count: "$SLURM_ARRAY_TASK_COUNT"
echo
echo Nodes: "$SLURM_NNODES"
echo Head node: "$SLURMD_NODENAME"
echo hostname "$(hostname)"
echo Nodelist: "$SLURM_NODELIST"
echo Tasks per node: "$SLURM_NTASKS_PER_NODE"
echo GPUs per node: "$SLURM_GPUS_PER_NODE"
echo
echo "N_SHOTS_PER_JOB=${N_SHOTS_PER_JOB}"
echo "MODEL_ID=${MODEL_ID}"
if [[ -n "$CKPT_NAME" ]]; then echo "CKPT_NAME=${CKPT_NAME}"; fi

if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    export UNIQUE_IDENTIFIER="$SLURM_JOBID"
fi
echo "UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER"

JOB_DIR=/pscratch/sd/k/kevinsg/bes_ml_jobs/
mkdir -p "$JOB_DIR" || exit
cd "$JOB_DIR" || exit
echo "Job directory: $PWD"

export WANDB__SERVICE_WAIT=300

# ---------------------- SHOT LIST --------------------------
# full list of all shots
shot_array=(145384 145387 145388 145391 145410 145419 145425 145422 157373 \
            157303 157322 157372 157374 158076 157376 157323 157375 157377 200021 
            145420 145427 200635 203659 203663 203665 203667 203671 203672 159443 203660
            189189 189191 189199)

# Compute chunking for this array task
idx=${SLURM_ARRAY_TASK_ID:-0}
total_shots=${#shot_array[@]}
chunk_size=$N_SHOTS_PER_JOB
start=$(( idx * chunk_size ))
# length to take (cap at remaining shots)
remain=$(( total_shots - start ))
take=$(( remain > chunk_size ? chunk_size : (remain > 0 ? remain : 0) ))

# Helpful: tell the user the recommended array max
#   (ceil(total_shots / N) - 1)
recommended_max=$(( ( (total_shots + chunk_size - 1) / chunk_size ) - 1 ))
echo "Total shots: ${total_shots}, chunk_size: ${chunk_size}"
echo "This task idx=${idx} will take start=${start}, count=${take}"
echo "Recommended --array=0-${recommended_max}"

# If this task is beyond the end, exit cleanly (some schedulers over-provision)
if (( take <= 0 )); then
  echo "No shots assigned to this task index ${idx}. Exiting."
  exit 0
fi

# Slice the shots for this task
predict_shots_chunk=( "${shot_array[@]:start:take}" )

# Build a Python-style list string: "['145384','145385']"
PREDICT_SHOTS_STR=$( printf ",'%s'" "${predict_shots_chunk[@]}" )
PREDICT_SHOTS_STR="[${PREDICT_SHOTS_STR:1}]"
export PREDICT_SHOTS_STR

# Output filename includes model_id + task index for clarity
export PRED_FILE="predictions_${idx}.hdf5"

# ------------------ PYTHON DRIVER (heredoc) ----------------
PYTHON_SCRIPT=$(cat << 'END_PY'
import sys, os, time, glob
from pathlib import Path
from datetime import timedelta

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import wandb
import numpy as np

from bes_ml2.main import BES_Trainer  # if needed elsewhere
from bes_ml2 import confinement_datamodule_4
from bes_ml2 import elm_lightning_model

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
num_nodes = int(os.getenv('SLURM_NNODES', '1'))
world_size = int(os.getenv('SLURM_NTASKS', 0))
world_rank = int(os.getenv('SLURM_PROCID', 0))
local_rank = int(os.getenv('SLURM_LOCALID', 0))
node_rank = int(os.getenv('SLURM_NODEID', 0))
print(f'World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})')

is_global_zero = (world_rank == 0)
if not is_global_zero:
    f = open(os.devnull, 'w'); sys.stdout = f

try:
    t_start = time.time()
    trial_name = f'{logger_hash}'
    experiment_dir = Path('./exp_gill01')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = experiment_dir.name
    experiment_parent_dir = experiment_dir.parent

    # Loggers
    tb_logger = TensorBoardLogger(
        save_dir=experiment_parent_dir,
        name=experiment_name,
        version=trial_name,
        default_hp_metric=False,
    )
    trial_dir = Path(tb_logger.log_dir).absolute()
    print(f"Trial directory: {trial_dir}")
    loggers = [tb_logger]

    # -------- model_id & checkpoint handling --------
    model_id = os.getenv('MODEL_ID', '41825238')
    ckpt_name_env = os.getenv('CKPT_NAME', '')
    ckpt_dir = Path(f'/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/{model_id}/checkpoints')
    if ckpt_name_env:
        checkpoint = ckpt_dir / ckpt_name_env
    else:
        # Prefer a checkpoint that looks like "epoch=...step=..."
        best_ckpts = sorted(
            [p for p in ckpt_dir.glob("*.ckpt") if "epoch=" in p.name and "step=" in p.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if best_ckpts:
            checkpoint = best_ckpts[0]
        else:
            # fallback to last.ckpt
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                checkpoint = last_ckpt
            else:
                raise FileNotFoundError(f"No suitable checkpoint found in {ckpt_dir}")
    print(f"Using checkpoint: {checkpoint}")

    # Load model
    lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(checkpoint_path=str(checkpoint))

    # Reuse trained window size
    W_model = int(getattr(lightning_model.hparams, "signal_window_size", getattr(lightning_model, "signal_window_size", 0)))

    # Prediction output path
    pred_file = os.getenv('PRED_FILE', f'predictions_{model_id}.hdf5')
    lightning_model.prediction_directory = f'/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/{model_id}/{pred_file}'
    lightning_model.log_dir = str(experiment_dir)

    # Data
    world_size = int(os.getenv('SLURM_NTASKS', 0))
    label_filter={3,4}
    datamodule = confinement_datamodule_4.Confinement_Datamodule(
        data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20250824_confinement_final.hdf5',
        seed=2,
        # lower_cutoff_frequency_hz=2.5e3,
        # upper_cutoff_frequency_hz=200e3,
        target_sampling_hz=250e3,
        standardize_signals=False,
        signal_window_size=32,
        predict_window_stride=16,
        n_rows=6,
        n_cols=4,
        label_filter=label_filter,
        num_classes=len(label_filter),
        fraction_test=0.22,
        fraction_validation=0.1,
        batch_size=256,
        num_workers=4,
        plot_data_stats=False,
        world_size=world_size,
        one_hot_labels=True,
        # force_validation_shots=['145384', '203659'],
        # force_test_shots=['145388', '145422'],
        predict_shots=eval(os.getenv('PREDICT_SHOTS_STR', '[]')),
    )
    datamodule.setup(stage='predict')

    # WandB (optional)
    wandb_log = False
    if wandb_log:
        wandb.login()
        wandb_logger = WandbLogger(save_dir=experiment_dir, project=experiment_name, name=trial_name)
        wandb_logger.watch(lightning_model, log='all', log_freq=100)
        loggers.append(wandb_logger)

    precision = 32
    frontends_active = list(lightning_model.frontends_active.values())
    some_unused = (False in frontends_active)

    trainer = Trainer(
        logger=loggers,
        num_nodes=num_nodes,
        precision=precision,
        strategy=DDPStrategy(find_unused_parameters=some_unused, timeout=timedelta(seconds=9600)),
    )
    trainer.predict(model=lightning_model, datamodule=datamodule)

    print(f'Python elapsed time {(time.time()-t_start)/60:.1f} min')
except Exception as e:
    print(f"An error occurred: {e}")
    if not is_global_zero:
        f.close(); sys.stdout = sys.__stdout__
    raise
finally:
    if not is_global_zero:
        f.close(); sys.stdout = sys.__stdout__
END_PY
)

echo "Script:"
echo "${PYTHON_SCRIPT}"

START_TIME=$(date +%s)
# Export MODEL_ID/CKPT_NAME to Python
export MODEL_ID
export CKPT_NAME
srun python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo "Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s"

exit $EXIT_CODE