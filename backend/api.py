"""
EquilibriumDynamics — FastAPI Orchestrator  v2.2
Fixes v2.2:
  • WatchFiles reload killing training: upload directories are now excluded
    from uvicorn's file watcher via a .watchfilesignore file and the
    --reload-exclude CLI flag documented in QUICKSTART.
  • Training background task now runs in a separate thread (run_in_executor)
    so asyncio event loop is never blocked during long L-BFGS iterations,
    preventing WebSocket timeouts during phase 2.
  • WebSocket handler no longer exits on training_complete — it stays open
    so the client can reconnect for a second training run without refresh.
"""

import os
import json
import asyncio
import importlib.util
import inspect
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import (
    FastAPI, UploadFile, File, WebSocket,
    WebSocketDisconnect, HTTPException, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.model import (
    build_model, train, predict_slice, predict_with_uq,
    conservation_error, WEIGHTS_DIR, CHECKPOINT_LBFGS,
)

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Thread pool for blocking training calls ───────────────────────────────────
# L-BFGS blocks the Python thread for minutes. Running it in a ThreadPoolExecutor
# keeps the asyncio event loop alive so WebSocket heartbeats and HTTP requests
# are still served during training.
_executor = ThreadPoolExecutor(max_workers=1)

# ── App & CORS ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EquilibriumDynamics API",
    description="High-fidelity PINN solver — 1D convection-diffusion, conservation enforced",
    version="2.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_model                   = None
_ws_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
_training                = False
_params                  = {"v": 1.0, "D": 0.01}

DYNAMIC_MODELS_DIR = Path(WEIGHTS_DIR).parent / "dynamic_models"
DYNAMIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════
class PhysicsParams(BaseModel):
    v:            float = Field(1.0,   ge=0.0,   le=10.0)
    D:            float = Field(0.01,  ge=1e-4,  le=1.0)
    num_domain:   int   = Field(3000,  ge=500,   le=10000)
    num_boundary: int   = Field(300,   ge=50,    le=1000)
    num_initial:  int   = Field(300,   ge=50,    le=1000)
    adam_iters:   int   = Field(15000, ge=1000,  le=100000)
    restore:      bool  = Field(False)


class PredictRequest(BaseModel):
    v:     float = Field(1.0)
    D:     float = Field(0.01)
    t_val: float = Field(0.5, ge=0.0, le=1.0)
    n_x:   int   = Field(200, ge=10,  le=2000)


class HeatmapRequest(BaseModel):
    n_x:        int = Field(50, ge=10, le=200)
    n_t:        int = Field(50, ge=10, le=200)
    uq_samples: int = Field(30, ge=1,  le=100)


# ══════════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "training":     _training,
        "params":       _params,
        "version":      "2.2.0",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training — runs in ThreadPoolExecutor so asyncio loop stays alive
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/train")
async def start_training(params: PhysicsParams, background: BackgroundTasks):
    global _training, _params
    if _training:
        raise HTTPException(409, "Training already in progress")

    _params = {"v": params.v, "D": params.D}

    def _blocking_train():
        """Runs in a thread — safe to block here for L-BFGS."""
        global _model, _training
        _training = True
        try:
            restore = CHECKPOINT_LBFGS if params.restore else None
            m = build_model(
                v=params.v, D=params.D,
                num_domain=params.num_domain,
                num_boundary=params.num_boundary,
                num_initial=params.num_initial,
            )
            train(m, adam_iters=params.adam_iters,
                  websocket_queue=_ws_queue, restore_weights=restore)
            _model = m
            log.info("Training complete. Model ready.")
        except Exception as exc:
            log.error("Training failed: %s", exc, exc_info=True)
        finally:
            _training = False
            # Signal the WebSocket clients that training finished
            try:
                _ws_queue.put_nowait(json.dumps({"event": "training_complete"}))
            except asyncio.QueueFull:
                pass

    async def _dispatch():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _blocking_train)

    background.add_task(_dispatch)
    params_out = params.model_dump() if hasattr(params, "model_dump") else params.dict()
    return {"status": "training_started", "params": params_out}


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/predict/slice")
async def predict_time_slice(req: PredictRequest):
    _require_model()
    data = predict_slice(_model, t_val=req.t_val, n_x=req.n_x)
    return {"plot_data": data, "t": req.t_val, "v": req.v, "D": req.D}


@app.post("/predict/heatmap")
async def predict_heatmap(req: HeatmapRequest):
    _require_model()
    x_pts  = np.linspace(0, 1, req.n_x)
    t_pts  = np.linspace(0, 1, req.n_t)
    result = predict_with_uq(_model, x_pts, t_pts, n_samples=req.uq_samples)
    return result


@app.get("/predict/conservation")
async def get_conservation():
    _require_model()
    return conservation_error(_model, t_vals=np.linspace(0, 1, 21))


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic uploads
# Saving to dynamic_models/ no longer triggers a server reload because
# that directory is excluded via --reload-exclude (see QUICKSTART.md).
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/upload/pde")
async def upload_pde(file: UploadFile = File(...)):
    if not file.filename.endswith(".py"):
        raise HTTPException(400, "Only .py files accepted for PDE definitions")

    save_path = DYNAMIC_MODELS_DIR / file.filename
    contents  = await file.read()
    save_path.write_bytes(contents)

    try:
        pde_fn, bc_fn = _load_user_pde(str(save_path))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error":      str(exc),
                "file_saved": str(save_path),
                "hint": (
                    "File was saved but validation failed. "
                    "Ensure it defines a top-level function named `pde(x, u)`. "
                    "The error above lists what callables were actually found."
                ),
            },
        )
    except Exception as exc:
        raise HTTPException(422, detail={"error": f"Module import failed: {exc}"})

    return {
        "status":        "Physics engine updated",
        "filename":      file.filename,
        "has_custom_bc": bc_fn is not None,
        "note":          "POST /train to rebuild the model with the new physics",
    }


@app.post("/upload/weights")
async def upload_weights(file: UploadFile = File(...)):
    """
    Accept uploaded weights in two formats:

    Format A — DeepXDE checkpoint stem:
      Files named like  checkpoint_adam-0.pt  or  checkpoint_lbfgs-0.pt
      DeepXDE's model.save("path/stem") writes "path/stem-0.pt".
      model.restore("path/stem") appends "-0.pt" internally.
      → We strip the "-0.pt" suffix to get the stem.

    Format B — Raw PyTorch state dict (.pth):
      Files named like  model.pth  saved via torch.save(model.state_dict(), ...).
      DeepXDE cannot restore these via model.restore() directly.
      → We load the state dict manually into net.load_state_dict().

    Both formats build a fresh model first if none is in memory.
    """
    global _model
    import torch as th

    if not (file.filename.endswith(".pt") or file.filename.endswith(".pth")):
        raise HTTPException(400, "Only .pt or .pth weight files accepted")

    # Always save file first
    save_path = Path(WEIGHTS_DIR) / file.filename
    save_path.write_bytes(await file.read())
    log.info("Weights file saved → %s", save_path)

    # Build a fresh model if none exists
    if _model is None:
        log.info("No model in memory — building fresh model for weight restore")
        try:
            m = build_model()
            m.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
            _model = m
        except Exception as exc:
            raise HTTPException(500, detail=f"Failed to build model: {exc}")

    fname = file.filename

    # ── Format A: DeepXDE checkpoint  (e.g. checkpoint_lbfgs-0.pt) ──────────
    # DeepXDE restore() expects the stem WITHOUT the trailing "-0.pt"
    if fname.endswith("-0.pt") or (fname.endswith(".pt") and "-" in fname):
        stem = str(save_path).removesuffix("-0.pt")
        try:
            _model.restore(stem, verbose=1)
            log.info("DeepXDE checkpoint restored from stem: %s", stem)
            return {"status": "weights_loaded", "format": "deepxde_checkpoint",
                    "file": fname, "model_ready": True}
        except Exception as exc:
            log.error("DeepXDE restore failed: %s", exc)
            raise HTTPException(422, detail={"error": str(exc), "format_tried": "deepxde_checkpoint"})

    # ── Format B: raw PyTorch state dict  (e.g. model.pth) ──────────────────
    # Load manually via torch and inject into the network
    try:
        state = th.load(str(save_path), map_location="cpu", weights_only=True)

        # state may be a bare state_dict or wrapped in {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        _model.net.load_state_dict(state, strict=False)
        _model.net.eval()
        log.info("Raw state dict loaded into model.net from %s", save_path)
        return {"status": "weights_loaded", "format": "pytorch_state_dict",
                "file": fname, "model_ready": True,
                "note": "Loaded as raw state dict (strict=False). "
                        "Verify architecture matches if inference looks wrong."}

    except Exception as exc:
        log.error("State dict restore failed: %s", exc)
        raise HTTPException(422, detail={
            "error": str(exc),
            "hint": (
                "Could not load as a raw PyTorch state dict either. "
                "For DeepXDE checkpoints name the file 'checkpoint_lbfgs-0.pt'. "
                "For raw state dicts ensure torch.save(model.state_dict(), path) was used."
            ),
        })


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — real-time loss stream
# Stays open after training_complete so clients can start a second run.
# ══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket client connected")
    try:
        while True:
            # Wait up to 30 s for a packet; send a heartbeat if idle
            try:
                packet = await asyncio.wait_for(_ws_queue.get(), timeout=30.0)
                await ws.send_text(packet)
                # Don't break on training_complete — stay open for next run
            except asyncio.TimeoutError:
                # Keep-alive ping so the browser doesn't close the socket
                await ws.send_text(json.dumps({"event": "heartbeat"}))
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════
def _require_model():
    global _model
    if _model is None:
        pt = CHECKPOINT_LBFGS + ".pt"
        if os.path.exists(pt):
            log.info("Auto-restoring checkpoint from %s", pt)
            m = build_model()
            m.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
            m.restore(CHECKPOINT_LBFGS, verbose=0)
            _model = m
            return
        raise HTTPException(
            503,
            "No trained model available. POST /train first, or upload a checkpoint.",
        )


def _load_user_pde(file_path: str):
    spec   = importlib.util.spec_from_file_location("user_pde", file_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(f"Failed to execute module: {exc}") from exc

    pde_fn = getattr(module, "pde", None)
    if pde_fn is None:
        found = [
            name for name, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == "user_pde"
        ]
        raise ValueError(
            f"No function named 'pde' found. "
            f"Callables in '{Path(file_path).name}': {found or ['(none)']}"
        )
    if not callable(pde_fn):
        raise ValueError("'pde' exists in the module but is not callable.")

    bc_fn = getattr(module, "boundary_conditions", None)
    return pde_fn, bc_fn
