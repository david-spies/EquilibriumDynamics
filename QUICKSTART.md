# EquilibriumDynamics — Quickstart

> High-fidelity PINN solver for the 1D convection-diffusion equation with enforced conservation of mass.

## Directory structure

```
EquilibriumDynamics/
├── backend/
│   ├── model.py            # PINN physics + training pipeline
│   ├── api.py              # FastAPI orchestrator + WebSocket
│   ├── dynamic_models/     # Hot-swapped .py PDE uploads land here
│   └── weights/            # Checkpoints: checkpoint_adam.pt, checkpoint_lbfgs.pt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       └── Dashboard.jsx
├── .watchfilesignore       # Prevents reload when uploading files
├── requirements.txt
└── QUICKSTART.md
```

## Installation

```bash
# 1. Clone
git clone https://github.com/your-username/EquilibriumDynamics.git
cd EquilibriumDynamics

# 2. Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Frontend dependencies
cd frontend
npm install
cd ..
```

## Running

### Step 1 — (Optional) Pre-train via CLI
Run this once to generate saved checkpoints before starting the API.
Press Ctrl+C after the Adam phase saves — L-BFGS can run for a very long
time and the Adam checkpoint alone is sufficient for inference.

```bash
python -m backend.model
```

Checkpoints saved to `backend/weights/checkpoint_adam.pt` and `checkpoint_lbfgs.pt`.

### Step 2 — Start the API
Use `--reload-exclude` to prevent uvicorn from restarting the server
when files are uploaded into `dynamic_models/` or `weights/`.

```bash
uvicorn backend.api:app \
  --reload \
  --reload-exclude "backend/dynamic_models/*" \
  --reload-exclude "backend/weights/*" \
  --port 8000
```

> **Why this matters:** Without `--reload-exclude`, dropping a `pde.py` or
> `.pt` file into those directories triggers a full server restart, which
> kills any running training task mid-epoch.

### Step 3 — Launch the dashboard

```bash
cd frontend
npm run dev        # opens http://localhost:5173
```

## Understanding the terminal output

| Message | Meaning | Action needed |
|---|---|---|
| `KeyboardInterrupt` traceback after Ctrl+C | Normal — Python prints the call stack when interrupted | None — checkpoint was already saved |
| `Adam checkpoint saved → backend/weights/checkpoint_adam` | Phase 1 complete | Safe to Ctrl+C here if L-BFGS is taking too long |
| `WatchFiles detected changes … Reloading` | A file in a watched directory changed | Use the `--reload-exclude` flags above to prevent this |
| `epoch: NNNN, loss: X.XX` stopping/lagging | Server reloaded mid-training due to WatchFiles | Restart with `--reload-exclude` flags |
| L-BFGS running for a long time with no output | Normal — L-BFGS is a batch optimizer, no per-step logging | Wait, or Ctrl+C (Adam checkpoint is still usable) |

## Plug-and-Play usage

| Action | How |
|---|---|
| Custom PDE | Write `pde(x, u)` in a `.py` file → drag into the Physics dropzone |
| Load weights | Drag a `.pt` checkpoint into the Weights dropzone |
| Warm start | Tick "Warm-start from checkpoint" before clicking Train |
| Watch convergence | WebSocket streams loss every 25 Adam iterations to the live chart |

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Model status, training state, version |
| `POST /train` | POST | Build & train the PINN (runs in background thread) |
| `POST /predict/slice` | POST | u(x, t) spatial slice |
| `POST /predict/heatmap` | POST | Full u(x,t) field + UQ bands |
| `GET /predict/conservation` | GET | Mass conservation diagnostic |
| `POST /upload/pde` | POST | Hot-swap physics equation |
| `POST /upload/weights` | POST | Load pre-trained weights |
| `WS /ws/training` | WS | Live loss stream + 30s keepalive heartbeat |

## Physics notes

The governing equation is:

```
∂u/∂t + v·∂u/∂x − D·∂²u/∂x² = 0
```

- Initial condition: `u(x, 0) = sin(π·x)`
- Boundary conditions: `u(0, t) = u(1, t) = 0`
- Conservation verified by tracking ∫₀¹ u(x, t) dx vs. t

The Péclet number **Pe = v/D** governs regime difficulty:

| Pe | Regime | Notes |
|---|---|---|
| < 1 | Diffusion-dominated | Smooth solutions, fast convergence |
| 1–10 | Mixed | Typical research range |
| > 10 | Convection-dominated | Sharp fronts, increase `num_domain` |
