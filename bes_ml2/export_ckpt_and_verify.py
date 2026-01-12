# # export_ckpt_and_verify.py
# import torch
# from bes_ml2 import elm_lightning_model

# ckpt_path = "/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/42078533/checkpoints/epoch=4-step=57120.ckpt"
# out_path  = ckpt_path.replace(".ckpt", "_weights.pt")

# # Force CPU so the load doesn't try to allocate on CUDA
# model = elm_lightning_model.Lightning_Model.load_from_checkpoint(
#     checkpoint_path=ckpt_path,
#     map_location="cpu",          # <<< key line
# )

# # Save only model weights in plain PyTorch format (CPU tensors)
# torch.save(model.state_dict(), out_path)
# print("Exported weights to:", out_path)

# export_ckpt_and_verify.py
import torch, numpy as np
from pathlib import Path
from typing import Any, Tuple
from bes_ml2 import elm_lightning_model

# ---------------- config ----------------
ckpt_path = "/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/42078533/checkpoints/epoch=4-step=57120.ckpt"
out_path  = ckpt_path.replace(".ckpt", "_weights.pt")

# ------------- helpers ------------------
def _sample_paths(ckpt: Path) -> Tuple[Path, Path, Path]:
    base = ckpt.parent / (ckpt.stem + "_sample_inputs")
    npz = base.with_suffix(".npz")
    pt  = base.with_suffix(".pt")
    raw = base.parent / (base.name + "_raw_batch.pt")
    return npz, pt, raw

def _torch_load_any(path: Path) -> Any:
    # Try default (2.6 uses weights_only=True), then weights_only=False,
    # then allowlist the offending numpy scalar if present.
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e1:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e2:
            try:
                from torch.serialization import safe_globals
                with safe_globals([np.core.multiarray.scalar]):
                    return torch.load(path, map_location="cpu")
            except Exception as e3:
                raise RuntimeError(
                    f"torch.load failed for {path}\n"
                    f"  default error: {e1}\n"
                    f"  weights_only=False error: {e2}\n"
                    f"  safe_globals allowlist error: {e3}\n"
                )

def _load_sample_input(ckpt: Path) -> Tuple[torch.Tensor, str]:
    """Return (x_example, source_str)."""
    npz_path, pt_path, _ = _sample_paths(ckpt)
    if npz_path.exists():
        arrs = np.load(str(npz_path))
        if "x_example" in arrs:
            x = torch.from_numpy(arrs["x_example"]).float()
            return x, f"NPZ:{npz_path.name}"
    if pt_path.exists():
        obj = _torch_load_any(pt_path)
        if isinstance(obj, dict) and "x_example" in obj:
            return obj["x_example"].detach().cpu().float(), f"PT:{pt_path.name}"
    raise FileNotFoundError(
        "Could not find sample inputs. Run your predict script to create *_sample_inputs.{npz,pt} first."
    )

def _load_raw_batch(ckpt: Path) -> Any | None:
    _, _, raw_path = _sample_paths(ckpt)
    if not raw_path.exists():
        return None
    try:
        obj = _torch_load_any(raw_path)
    except Exception as e:
        print(f"[warn] Could not load raw batch ({raw_path.name}): {e}")
        return None
    if isinstance(obj, dict) and "raw_first_batch" in obj:
        return obj["raw_first_batch"]
    return obj

def _first_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None

def _describe(obj: Any) -> str:
    t = _first_tensor(obj)
    if t is None:
        return f"{type(obj)}"
    t = t.detach().cpu()
    mn = t.min().item() if t.numel() else float("nan")
    mx = t.max().item() if t.numel() else float("nan")
    return f"Tensor(shape={tuple(t.shape)}, dtype={t.dtype}, min={mn:.4g}, max={mx:.4g})"

def _run_inference(model, x_example: torch.Tensor, raw_batch: Any | None):
    """
    Try model(x_example) first. If that fails and raw_batch is available,
    try model.predict_step(raw_batch, 0, 0),
    then forward on the first tensor inside raw_batch.
    """
    model.eval()
    with torch.no_grad():
        # Preferred path: direct forward on the example tensor we saved
        try:
            y = model(x_example)
            return y, "forward(x_example)"
        except Exception as e1:
            if raw_batch is not None:
                try:
                    y = model.predict_step(raw_batch, 0, 0)
                    return y, "predict_step(raw_batch)"
                except Exception as e2:
                    xb = _first_tensor(raw_batch)
                    if xb is not None:
                        y = model(xb.detach().cpu().float())
                        return y, "forward(first_tensor(raw_batch))"
                    raise RuntimeError(
                        f"Could not run inference.\n"
                        f"forward(x_example) error: {e1}\n"
                        f"predict_step(raw_batch) error: {e2}"
                    )
            else:
                raise RuntimeError(f"Could not run forward(x_example): {e1}")

def _assert_finite(obj: Any, label: str):
    t = _first_tensor(obj)
    if t is None:
        raise AssertionError(f"{label}: No tensor found in output.")
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise AssertionError(f"{label}: Output contains non-finite values (count={bad}).")

# ---------------- main ------------------
def main():
    ckpt = Path(ckpt_path)

    # 1) Load model on CPU from checkpoint
    model = elm_lightning_model.Lightning_Model.load_from_checkpoint(
        checkpoint_path=str(ckpt),
        map_location="cpu",
    )

    # 2) Load sample inputs + optional raw batch
    x_example, src = _load_sample_input(ckpt)
    raw_batch = _load_raw_batch(ckpt)

    print(f"[verify] Loaded sample input from {src}: shape={tuple(x_example.shape)}, dtype={x_example.dtype}")
    if raw_batch is not None:
        print(f"[verify] Raw batch available. First tensor: {_describe(raw_batch)}")
    else:
        print("[verify] Raw batch not available; verifying via x_example only.")

    # 3) Forward once and validate
    out_obj, method = _run_inference(model, x_example, raw_batch)
    print(f"[verify] Inference via {method}: {_describe(out_obj)}")
    _assert_finite(out_obj, "checkpoint-forward")
    print("[verify] Output is finite ✅")

    # 4) Export state_dict
    torch.save(model.state_dict(), out_path)
    print("Exported weights to:", out_path)

    # 5) Reload into a fresh model and confirm outputs match
    fresh = elm_lightning_model.Lightning_Model.load_from_checkpoint(
        checkpoint_path=str(ckpt),
        map_location="cpu",
    )
    sd = torch.load(out_path, map_location="cpu")  # pure state_dict
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict reports missing={missing}, unexpected={unexpected}")

    fresh_out, fresh_method = _run_inference(fresh, x_example, raw_batch)
    print(f"[verify] Fresh model via {fresh_method}: {_describe(fresh_out)}")
    _assert_finite(fresh_out, "fresh-forward")

    # 6) Numerical consistency check (on first tensor)
    y1 = _first_tensor(out_obj).detach().cpu()
    y2 = _first_tensor(fresh_out).detach().cpu()
    same_shape = y1.shape == y2.shape
    close = torch.allclose(y1, y2, rtol=1e-3, atol=1e-5)
    max_abs = (y1 - y2).abs().max().item() if same_shape else float("nan")
    print(f"[verify] Reload parity: shape_equal={same_shape}, allclose={close}, max_abs_diff={max_abs:.3e}")
    if not same_shape or not close:
        print("[warn] Outputs differ beyond tolerance. This can happen if the checkpoint stores "
              "non-state config/buffers or if there is nondeterminism in predict paths.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
