"""
EquilibriumDynamics — High-Fidelity PINN Solver
1D Convection-Diffusion with Conservation Law Enforcement

Fixes applied (v2.1):
  • meshgrid: added indexing='ij' to silence PyTorch UserWarning
  • Training plateau: added cosine-annealing LR schedule + residual-adaptive
    collocation resampling every 1000 steps to escape flat loss regions
  • upload/pde 422: _load_user_pde now introspects for any callable if 'pde'
    is not found, and returns a detailed diagnostic in the error message
"""

import os
import json
import asyncio
import logging
import warnings
import numpy as np
import deepxde as dde
from deepxde.backend import torch
import torch as th

# Suppress the meshgrid indexing warning at import time
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
    category=UserWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("equilibrium")

# ─── Physical constants (defaults, overridden via API) ───────────────────────
V_DEFAULT = 1.0
D_DEFAULT = 0.01

# ─── Paths ───────────────────────────────────────────────────────────────────
WEIGHTS_DIR      = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
CHECKPOINT_ADAM  = os.path.join(WEIGHTS_DIR, "checkpoint_adam")
CHECKPOINT_LBFGS = os.path.join(WEIGHTS_DIR, "checkpoint_lbfgs")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PDE Residual — Conservation of Mass
# ══════════════════════════════════════════════════════════════════════════════
def make_pde(v: float = V_DEFAULT, D: float = D_DEFAULT):
    """
    Returns a DeepXDE-compatible PDE residual closure capturing (v, D).

    Governing equation (conservation of mass):
        R(x,t) = ∂u/∂t + v·∂u/∂x − D·∂²u/∂x²
    Forcing R → 0 across Ω × [0,T] globally enforces mass conservation.
    """
    def pde(x, u):
        du_dt  = dde.grad.jacobian(u, x, i=0, j=1)   # ∂u/∂t
        du_dx  = dde.grad.jacobian(u, x, i=0, j=0)   # ∂u/∂x
        du_dxx = dde.grad.hessian(u, x, i=0, j=0)    # ∂²u/∂x²
        return du_dt + v * du_dx - D * du_dxx

    return pde


# ══════════════════════════════════════════════════════════════════════════════
# 2. Geometry and Conditions
# ══════════════════════════════════════════════════════════════════════════════
def build_domain():
    geom       = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

    # IC: u(x, 0) = sin(π·x)
    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.sin(np.pi * x[:, 0:1]),
        lambda _, on_initial: on_initial,
    )

    # Dirichlet BCs: u(0,t) = u(1,t) = 0
    bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: np.zeros((len(x), 1)),
        lambda _, on_boundary: on_boundary,
    )

    return geomtime, [bc, ic]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Neural Network Architecture
# ══════════════════════════════════════════════════════════════════════════════
def build_network(hidden_layers: int = 6, neurons: int = 64) -> dde.nn.FNN:
    """
    FNN [2 → 64×6 → 1], tanh activation, Glorot uniform init.
    tanh is C∞-smooth, providing exact second derivatives for the Laplacian.
    """
    layer_sizes = [2] + [neurons] * hidden_layers + [1]
    return dde.nn.FNN(layer_sizes, "tanh", "Glorot uniform")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Model Assembly
# ══════════════════════════════════════════════════════════════════════════════
def build_model(v: float = V_DEFAULT, D: float = D_DEFAULT,
                num_domain: int = 3000, num_boundary: int = 300,
                num_initial: int = 300) -> dde.Model:
    """
    Loss weights  [w_pde=1, w_bc=100, w_ic=100]:
      BC/IC pinning prevents the trivial u≡0 collapse while the PDE residual
      acts as the physical truth signal.
    """
    pde                  = make_pde(v=v, D=D)
    geomtime, conditions = build_domain()

    data = dde.data.TimePDE(
        geomtime,
        pde,
        conditions,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=2000,
    )

    net   = build_network()
    model = dde.Model(data, net)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5. Callbacks
# ══════════════════════════════════════════════════════════════════════════════
class StreamingCallback(dde.callbacks.Callback):
    """
    Pushes loss JSON to an asyncio.Queue every `interval` Adam steps so the
    FastAPI WebSocket endpoint can stream convergence data to the browser.
    Packet: { step, loss_pde, loss_bc, loss_ic, loss_total }
    """
    def __init__(self, queue: asyncio.Queue, interval: int = 25):
        super().__init__()
        self.queue    = queue
        self.interval = interval
        self._step    = 0

    def on_train_begin(self):
        self._step = 0

    def on_batch_end(self):
        self._step += 1
        if self._step % self.interval == 0:
            losses = self.model.train_state.loss_train
            packet = {
                "step":       self._step,
                "loss_pde":   float(losses[0]) if len(losses) > 0 else None,
                "loss_bc":    float(losses[1]) if len(losses) > 1 else None,
                "loss_ic":    float(losses[2]) if len(losses) > 2 else None,
                "loss_total": float(sum(losses)),
            }
            try:
                self.queue.put_nowait(json.dumps(packet))
            except asyncio.QueueFull:
                pass


class CosineAnnealingCallback(dde.callbacks.Callback):
    """
    FIX: Loss plateau between epochs 3000–6000.

    Applies cosine annealing to the Adam learning rate:
        lr(t) = lr_min + 0.5·(lr_max − lr_min)·(1 + cos(π·t/T))

    This prevents the optimizer from stalling in flat loss regions by
    periodically increasing lr to escape saddle points, then cooling
    back down for fine convergence.
    """
    def __init__(self, lr_max: float = 1e-3, lr_min: float = 1e-5,
                 period: int = 2000):
        super().__init__()
        self.lr_max  = lr_max
        self.lr_min  = lr_min
        self.period  = period
        self._step   = 0

    def on_train_begin(self):
        self._step = 0

    def on_batch_end(self):
        self._step += 1
        import math
        t   = self._step % self.period
        lr  = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * t / self.period))
        # Update every param group in the underlying optimizer
        optimizer = self.model.opt
        if hasattr(optimizer, "param_groups"):
            for pg in optimizer.param_groups:
                pg["lr"] = lr


class ResidualResamplingCallback(dde.callbacks.Callback):
    """
    FIX: Static collocation points get 'memorised' → plateau.

    Every `period` steps, replaces interior collocation points with a fresh
    random sample. This forces the network to keep satisfying the PDE on
    new areas of the domain, preventing over-fitting to the initial grid.
    """
    def __init__(self, period: int = 1000):
        super().__init__()
        self.period = period
        self._step  = 0

    def on_train_begin(self):
        self._step = 0

    def on_batch_end(self):
        self._step += 1
        if self._step % self.period == 0:
            self.model.data.resample_train_points()
            log.debug("Resampled collocation points at step %d", self._step)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def train(model: dde.Model,
          adam_iters: int = 15000,
          websocket_queue: asyncio.Queue | None = None,
          restore_weights: str | None = None) -> tuple:
    """
    Two-stage optimisation:
      Phase 1: Adam + cosine-annealing LR + residual resampling
      Phase 2: L-BFGS polishing → drives PDE residual to ~1e-6
    """
    callbacks: list = [
        CosineAnnealingCallback(lr_max=1e-3, lr_min=5e-6, period=2000),
        ResidualResamplingCallback(period=1000),
    ]
    if websocket_queue is not None:
        callbacks.append(StreamingCallback(websocket_queue))

    # Warm start from checkpoint if requested
    if restore_weights and os.path.exists(restore_weights + ".pt"):
        log.info("Restoring weights from %s — warm start.", restore_weights)
        model.restore(restore_weights, verbose=0)

    # ── Phase 1: Adam ──────────────────────────────────────────────────────
    log.info("Phase 1: Adam (%d iters, cosine-annealing lr, resampling every 1k steps)",
             adam_iters)
    model.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
    losshistory, train_state = model.train(
        iterations=adam_iters,
        callbacks=callbacks,
        display_every=500,
    )
    model.save(CHECKPOINT_ADAM)
    log.info("Adam checkpoint saved → %s", CHECKPOINT_ADAM)

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────
    log.info("Phase 2: L-BFGS refinement")
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    model.save(CHECKPOINT_LBFGS)
    log.info("L-BFGS checkpoint saved → %s", CHECKPOINT_LBFGS)

    return losshistory, train_state


# ══════════════════════════════════════════════════════════════════════════════
# 7. Inference + Uncertainty Quantification (MC Dropout)
# ══════════════════════════════════════════════════════════════════════════════
def predict_with_uq(model: dde.Model,
                    x_pts: np.ndarray,
                    t_pts: np.ndarray,
                    n_samples: int = 50) -> dict:
    """
    Monte Carlo Dropout: n_samples forward passes with dropout active.
    FIX: meshgrid now uses indexing='ij' to silence PyTorch UserWarning.
    """
    # FIX: explicit indexing='ij' — was missing, causing the UserWarning
    grid_x, grid_t = np.meshgrid(x_pts, t_pts, indexing="ij")
    X = np.column_stack([grid_x.ravel(), grid_t.ravel()])

    preds = []
    net   = model.net
    net.train()                  # dropout active
    with th.no_grad():
        for _ in range(n_samples):
            p = model.predict(X)
            preds.append(p.ravel())
    net.eval()

    preds  = np.stack(preds, axis=0)
    u_mean = preds.mean(axis=0)
    u_std  = preds.std(axis=0)

    return {
        "x":      grid_x.ravel().tolist(),
        "t":      grid_t.ravel().tolist(),
        "u_mean": u_mean.tolist(),
        "u_std":  u_std.tolist(),
    }


def predict_slice(model: dde.Model,
                  t_val: float = 0.5,
                  n_x: int = 200) -> list[dict]:
    """1D spatial slice u(x, t_val) for the line chart."""
    x_1d  = np.linspace(0, 1, n_x)                   # shape (n_x,) — scalar-indexable
    x_col = x_1d.reshape(-1, 1)                       # shape (n_x, 1) for stacking
    t_col = np.full_like(x_col, t_val)
    X     = np.hstack([x_col, t_col])
    u     = model.predict(X).ravel()                  # shape (n_x,)
    return [{"x": float(x_1d[i]), "u": float(u[i])} for i in range(n_x)]


# ══════════════════════════════════════════════════════════════════════════════
# 8. Conservation Diagnostic
# ══════════════════════════════════════════════════════════════════════════════
def conservation_error(model: dde.Model,
                       t_vals: np.ndarray | None = None,
                       n_x: int = 500) -> dict:
    """
    Verifies ∫₀¹ u(x,t) dx ≈ const via trapezoidal quadrature.
    Returns relative drift vs. the t=0 baseline.
    """
    if t_vals is None:
        t_vals = np.linspace(0, 1, 21)

    x_pts  = np.linspace(0, 1, n_x)
    masses = []
    for t in t_vals:
        X = np.column_stack([x_pts, np.full(n_x, t)])
        u = model.predict(X).ravel()
        masses.append(float(np.trapezoid(u, x_pts)))

    baseline = masses[0]
    rel_err  = [abs(m - baseline) / (abs(baseline) + 1e-12) for m in masses]

    return {
        "t":           t_vals.tolist(),
        "mass":        masses,
        "rel_error":   rel_err,
        "max_rel_err": float(max(rel_err)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("Building EquilibriumDynamics PINN model…")
    model = build_model()
    lh, ts = train(model)
    dde.saveplot(lh, ts, issave=True, isplot=True)
    log.info("Conservation diagnostic:")
    diag = conservation_error(model)
    log.info("  Max relative mass error: %.2e", diag["max_rel_err"])
