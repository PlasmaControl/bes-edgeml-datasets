#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=2
#SBATCH --array=0
#SBATCH --time=00:30:00
#SBATCH --qos=debug

# ---------------------- USER CONTROLS ----------------------
# You can pass 2 positional args to this script:
#   $1 -> N_SHOTS_PER_JOB (default 2)
#   $2 -> MODEL_ID        (default 41825238)
# Or set env vars N_SHOTS_PER_JOB / MODEL_ID when calling sbatch.

N_SHOTS_PER_JOB="${1:-${N_SHOTS_PER_JOB:-2}}"
MODEL_ID="${2:-${MODEL_ID:-45481719}}"

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
shot_array=(145384 145420 145425 157303 157372 157375 158076 200643 203416 203420 203483 203664 203671 204292 204837 145388 145419 157322 157373 203417 203423 203665 203672 145387 145391 145410 145422 145427 157323 157374 157377 159443 189189 189191 189199 200634 200637 200638 200639 203152 203418 203419 203469 203470 203471 203484 203659 203660 203663 203667 203946 204286 204287 204288 204289 204290 204291 204294 204296 204297 204299 204301 204302 204303)

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

from bes_ml2.train_velo import BES_Trainer  # if needed elsewhere
from bes_ml2 import velocimetry_datamodule
from bes_ml2 import elm_lightning_model

import json, numpy as np, torch
from pathlib import Path
from collections.abc import Sequence

def _first_pred_loader(dm):
    loaders = dm.predict_dataloader()
    return loaders[0] if isinstance(loaders, Sequence) else loaders

def _find_model_tensor(batch):
    # Recursively find the first reasonably-shaped float tensor
    def walk(x):
        if torch.is_tensor(x) and x.ndim >= 3:
            return x
        if isinstance(x, dict):
            for v in x.values():
                t = walk(v)
                if t is not None: return t
        if isinstance(x, (list, tuple)):
            for v in x:
                t = walk(v)
                if t is not None: return t
        return None
    return walk(batch)

def save_sample_inputs_from_dm(datamodule, out_base_path: Path, W_model: int | None = None, max_batch_to_save: int = 4):
    loader = _first_pred_loader(datamodule)
    batch = next(iter(loader))  # first batch from predict
    x_batch = _find_model_tensor(batch)
    if x_batch is None:
        raise RuntimeError("Could not locate a tensor input in the first predict batch.")

    # Detach & put on CPU
    x_batch = x_batch.detach().cpu().float()
    x_small = x_batch[:max_batch_to_save]
    x_example = x_batch[:1].clone()

    base = out_base_path  # e.g., /.../epoch=4-step=57120_sample_inputs
    base.parent.mkdir(parents=True, exist_ok=True)

    # 1) Torch: easy for PyTorch users
    torch.save(
        {"x_example": x_example, "x_batch": x_small},
        base.with_suffix(".pt")
    )

    # 2) NumPy: portable for anyone
    np.savez_compressed(
        str(base.with_suffix(".npz")),
        x_example=x_example.numpy(),
        x_batch=x_small.numpy(),
    )

    # 3) Raw batch (exactly what the DataLoader yielded) for auditability
    torch.save({"raw_first_batch": batch}, base.parent / (base.name + "_raw_batch.pt"))

    # 4) Human-friendly metadata
    meta = {
        "example_shape": list(x_example.shape),
        "batch_shape": list(x_small.shape),
        "dtype": "float32",
        "notes": "Shapes are exactly as produced by your predict dataloader. "
                 "Typical is (B, 1, W, R, C).",
    }
    if W_model is not None:
        meta["signal_window_size_W"] = int(W_model)
    (base.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

    print(f"[samples] Saved: {base.with_suffix('.npz')}")
    print(f"[samples] Saved: {base.with_suffix('.pt')}")
    print(f"[samples] Saved: {(base.parent / (base.name + '_raw_batch.pt'))}")
    print(f"[samples] Saved: {base.with_suffix('.json')}")

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
num_nodes = int(os.getenv("SLURM_NNODES", "1"))
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
    world_size   = int(os.getenv('SLURM_NTASKS', 0))
    block_cols   = [1, 3, 5, 7]
    row_stride   = 1
    row_offset   = 4
    R_sel        = len(np.arange(8)[row_offset::row_stride])
    C_sel        = len(block_cols) 

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
        # 205830: [()],
    }

    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
            data_file='/global/cfs/cdirs/m3586/kgill/velocimetry_data/20251027_raw_signals_psi_interp.hdf5',
            signal_window_size=48,
            batch_size=256,
            num_workers=4,
            seed=0,
            world_size=world_size,
            standardize_signals=False,
            split_method='shot',
            train_shots = [
                # 145xxx (need some representation)
                '145384', '145420', '145425',
                # 157xxx (need some representation)
                '157303', '157372', '157375', '158076',
                # 200xxx
                '200643',
                # Early 203xxx (critical based on CV)
                '203416', '203420', '203483',
                # Late 203xxx + 204xxx
                '203664', '203671', '204292', '204837',
            ],
            validation_shots = [
                # Sample from each group for representative validation
                '145388', '145419',      # 145xxx
                '157322', '157373',      # 157xxx  
                '203417', '203423',      # early 203xxx
                '203665', '203672',      # late 203xxx
            ],            
            test_shots = ['145387', '145391', '145410', '145422', '145427', '157323', '157374', '157377', '159443', '189189', '189191', '189199', '200634', '200637', '200638', '200639', '203152', '203418', '203419', '203469', '203470', '203471', '203484', '203659', '203660', '203663', '203667', '203946', '204286', '204287', '204288', '204289', '204290', '204291', '204294',  '204296', '204297', '204299', '204301', '204302', '204303'],
            predict_shots=eval(os.getenv('PREDICT_SHOTS_STR', '[]')),
            split_train_data_per_gpu=False,
            do_flip_augmentation=True,
            shot_time_windows = good_times_psi_93,
            block_cols=block_cols,
            row_stride=row_stride,
            row_offset=row_offset,
            target_sampling_hz=1_000_000.0,
            label_target_psi=0.68,
            label_tolerance_ms=0.6,
            normalize_label_magnitude=False,
            window_hop=1,
            n_rows=R_sel,   # 8 with your settings
            n_cols=C_sel,   # 4 with ('last',4)
            predict_window_stride=48,
    )
    datamodule.setup(stage='predict')

    # Derive an output base next to the checkpoint you're already using
    samples_base = checkpoint.parent / (checkpoint.stem + "_sample_inputs")

    # Save a tiny batch and a single-example tensor
    save_sample_inputs_from_dm(datamodule, samples_base, W_model=W_model, max_batch_to_save=4)

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