"""
EquilibriumDynamics — Generate All Experiment Weights
======================================================
Run this ONCE on your machine to produce three trained checkpoint files
that can be dropped into the UI Weights dropzone for instant inference.

Usage:
    cd ~/EquilibriumDynamics
    source venv/bin/activate
    python experiments/generate_weights.py

Output files in backend/weights/:
    exp1_pde_dominant-0.pt      ← drop for Exp 1 (rebalanced loss weights)
    exp2_burgers-0.pt           ← drop for Exp 2 (Burgers' equation)
    exp3_fisher_kpp-0.pt        ← drop for Exp 3 (Fisher-KPP reaction)

Transfer learning chain (optional but recommended):
    Exp1 weights → warm-start Exp2 (same linear physics, just reweighted)
    Exp2 weights → warm-start Exp3 (nonlinear base → add reaction term)

Each experiment saves independently so you can stop mid-run and still
have usable checkpoints for the experiments that completed.
"""

import os
import sys
import time
import numpy as np
import deepxde as dde
import torch as th

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def ckpt(name):
    return os.path.join(WEIGHTS_DIR, name)

def make_base_domain(num_domain=3000, num_boundary=300, num_initial=300):
    geom = dde.geometry.Interval(0, 1)
    td   = dde.geometry.TimeDomain(0, 1)
    gt   = dde.geometry.GeometryXTime(geom, td)
    ic   = dde.icbc.IC(gt, lambda x: np.sin(np.pi * x[:, 0:1]),
                       lambda _, on_initial: on_initial)
    bc   = dde.icbc.DirichletBC(gt, lambda x: np.zeros((len(x), 1)),
                                lambda _, on_boundary: on_boundary)
    return gt, [bc, ic], num_domain, num_boundary, num_initial

def build_net():
    return dde.nn.FNN([2] + [64] * 6 + [1], "tanh", "Glorot uniform")

def two_phase_train(model, adam_iters, loss_weights, lr=1e-3):
    """Standard Adam → L-BFGS pipeline used across all experiments."""
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    model.train(iterations=adam_iters, display_every=2000)
    model.compile("L-BFGS")
    lh, ts = model.train()
    return lh, ts

def restore_into(model, stem):
    """Load a DeepXDE checkpoint stem into an already-compiled model."""
    full = stem + "-0.pt"
    if os.path.exists(full):
        model.restore(stem, verbose=1)
        print(f"  Warm-started from {full}")
        return True
    print(f"  No checkpoint at {full} — training from scratch")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Rebalanced loss weights  [PDE=100, BC=10, IC=10]
# Same 1D convection-diffusion as baseline, physics-dominant weighting.
# Expected: PDE residual < 1e-2 after L-BFGS.
# ══════════════════════════════════════════════════════════════════════════════
def run_exp1():
    print("\n" + "="*60)
    print("  EXP 1: Convection-Diffusion, PDE-dominant weights [100,10,10]")
    print("="*60)

    V, D = 1.0, 0.01

    def pde(x, u):
        return (dde.grad.jacobian(u, x, i=0, j=1)
                + V * dde.grad.jacobian(u, x, i=0, j=0)
                - D * dde.grad.hessian(u, x, i=0, j=0))

    gt, cond, nd, nb, ni = make_base_domain()
    data  = dde.data.TimePDE(gt, pde, cond, num_domain=nd,
                             num_boundary=nb, num_initial=ni, num_test=2000)
    net   = build_net()
    model = dde.Model(data, net)

    t0 = time.time()
    two_phase_train(model, adam_iters=15000, loss_weights=[100, 10, 10])
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")

    stem = ckpt("exp1_pde_dominant")
    model.save(stem)
    print(f"  Saved → {stem}-0.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Burgers' equation  (nonlinear self-advection)
# ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
# Warm-starts from Exp1 weights — shared tanh network shape allows transfer.
# The nonlinear u·∂u/∂x term causes wave steepening toward a near-shock.
# ══════════════════════════════════════════════════════════════════════════════
def run_exp2(exp1_model=None):
    print("\n" + "="*60)
    print("  EXP 2: Burgers' Equation  ν=0.01  (near-shock formation)")
    print("="*60)

    NU = 0.01

    def pde(x, u):
        return (dde.grad.jacobian(u, x, i=0, j=1)
                + u * dde.grad.jacobian(u, x, i=0, j=0)
                - NU * dde.grad.hessian(u, x, i=0, j=0))

    gt, cond, _, _, _ = make_base_domain()
    # More collocation points to resolve the near-shock region
    data  = dde.data.TimePDE(gt, pde, cond,
                             num_domain=5000, num_boundary=400,
                             num_initial=400, num_test=3000)
    net   = build_net()
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[100, 10, 10])

    # Transfer: copy Exp1 weights into Exp2 network
    if exp1_model is not None:
        try:
            state = exp1_model.net.state_dict()
            model.net.load_state_dict(state, strict=True)
            print("  Transfer learning: Exp1 weights loaded into Exp2 network")
        except Exception as e:
            print(f"  Transfer failed ({e}), training from scratch")
    else:
        restore_into(model, ckpt("exp1_pde_dominant"))

    t0 = time.time()
    two_phase_train(model, adam_iters=20000, loss_weights=[100, 10, 10])
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")

    stem = ckpt("exp2_burgers")
    model.save(stem)
    print(f"  Saved → {stem}-0.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Fisher-KPP reaction-diffusion
# ∂u/∂t + v·∂u/∂x − D·∂²u/∂x² = k·u·(1−u)
# Curriculum: Phase1 = smooth diffusion only → Phase2 = full nonlinear PDE
# Warm-starts from Exp1 weights (same linear base, then reaction added).
# Mass intentionally GROWS over time due to the logistic source term.
# ══════════════════════════════════════════════════════════════════════════════
def run_exp3(exp1_model=None):
    print("\n" + "="*60)
    print("  EXP 3: Fisher-KPP Reaction-Diffusion  k=1.0")
    print("  Curriculum: diffusion-only → full PDE")
    print("="*60)

    V, D, K = 1.0, 0.01, 1.0

    def pde_easy(x, u):
        """Curriculum Phase 1: linear diffusion only (no reaction, high D)."""
        return (dde.grad.jacobian(u, x, i=0, j=1)
                + V * dde.grad.jacobian(u, x, i=0, j=0)
                - 0.1 * dde.grad.hessian(u, x, i=0, j=0))

    def pde_full(x, u):
        """Full Fisher-KPP with reaction term."""
        return (dde.grad.jacobian(u, x, i=0, j=1)
                + V * dde.grad.jacobian(u, x, i=0, j=0)
                - D * dde.grad.hessian(u, x, i=0, j=0)
                - K * u * (1.0 - u))

    geom = dde.geometry.Interval(0, 1)
    td   = dde.geometry.TimeDomain(0, 1)
    gt   = dde.geometry.GeometryXTime(geom, td)

    # Half-amplitude IC to avoid u>1 saturation at t=0
    ic = dde.icbc.IC(gt, lambda x: 0.5 * np.sin(np.pi * x[:, 0:1]),
                     lambda _, on_initial: on_initial)
    bc = dde.icbc.DirichletBC(gt, lambda x: np.zeros((len(x), 1)),
                              lambda _, on_boundary: on_boundary)
    cond = [bc, ic]

    # ── Curriculum Phase 1: easy PDE ──────────────────────────────────
    data_easy = dde.data.TimePDE(gt, pde_easy, cond,
                                 num_domain=4000, num_boundary=400,
                                 num_initial=400, num_test=2000)
    net   = build_net()
    model = dde.Model(data_easy, net)
    model.compile("adam", lr=1e-3, loss_weights=[100, 10, 10])

    # Transfer from Exp1 if available
    if exp1_model is not None:
        try:
            model.net.load_state_dict(exp1_model.net.state_dict(), strict=True)
            print("  Transfer learning: Exp1 weights loaded into Exp3 network")
        except Exception as e:
            print(f"  Transfer failed ({e}), training from scratch")
    else:
        restore_into(model, ckpt("exp1_pde_dominant"))

    print("  Curriculum Phase 1: smooth diffusion (5000 iters)...")
    model.train(iterations=5000, display_every=2500)

    # ── Curriculum Phase 2: swap in full nonlinear PDE ────────────────
    print("  Curriculum Phase 2: full Fisher-KPP (10000 iters)...")
    data_full = dde.data.TimePDE(gt, pde_full, cond,
                                 num_domain=4000, num_boundary=400,
                                 num_initial=400, num_test=2000)
    model.data = data_full
    model.compile("adam", lr=5e-4, loss_weights=[100, 10, 10])

    t0 = time.time()
    model.train(iterations=10000, display_every=2500)

    # ── Phase 3: L-BFGS polish ────────────────────────────────────────
    model.compile("L-BFGS")
    model.train()
    print(f"  Training time: {(time.time()-t0)/60:.1f} min")

    stem = ckpt("exp3_fisher_kpp")
    model.save(stem)
    print(f"  Saved → {stem}-0.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main — run all three in sequence, passing models forward for transfer learning
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total_start = time.time()

    print("\nEquilibriumDynamics — Generating experiment weight files")
    print("Transfer learning chain: Exp1 → Exp2, Exp1 → Exp3")
    print(f"Output directory: {os.path.abspath(WEIGHTS_DIR)}\n")

    skip = set(sys.argv[1:])   # e.g. "python generate_weights.py 2 3" skips Exp1

    exp1_model = None
    exp2_model = None
    exp3_model = None

    if "1" not in skip:
        exp1_model = run_exp1()
    else:
        print("Skipping Exp1 (will load from checkpoint for transfer)")

    if "2" not in skip:
        exp2_model = run_exp2(exp1_model)

    if "3" not in skip:
        exp3_model = run_exp3(exp1_model)

    total = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  All experiments complete in {total:.1f} min")
    print(f"  Files in {os.path.abspath(WEIGHTS_DIR)}:")
    for f in sorted(os.listdir(WEIGHTS_DIR)):
        if f.endswith(".pt"):
            size = os.path.getsize(os.path.join(WEIGHTS_DIR, f)) / 1024
            print(f"    {f:45s} {size:7.1f} KB")
    print(f"\n  Drop any of these into the UI Weights dropzone for instant inference.")
    print(f"  Rename to end in .pth if the dropzone filter requires it.")
    print("="*60)
