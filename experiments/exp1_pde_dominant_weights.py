"""
EquilibriumDynamics — Experiment 1: Rebalanced Loss Weights
============================================================
Problem:  Same 1D convection-diffusion as baseline.
Change:   Flip the weighting philosophy.
          Baseline was [PDE=1, BC=100, IC=100] — BC/IC dominated.
          Here:        [PDE=100, BC=10, IC=10] — physics dominates.

Why:  At Adam step 15000, PDE loss was 14× larger than BC and 33× larger
      than IC. The network was fitting boundaries perfectly while ignoring
      the interior dynamics. Elevating PDE weight forces the optimizer to
      treat the conservation law as the primary objective.

Expected outcome:  PDE residual should drop below 1e-2 after L-BFGS.
                   BC/IC may be slightly less tight, but mass conservation
                   will be more physically meaningful.

Drop this file into the Physics dropzone, then click Train PINN.
"""

import numpy as np
import deepxde as dde

V = 1.0    # convection velocity
D = 0.01   # diffusion coefficient  →  Pe = v/D = 100


def pde(x, u):
    """1D convection-diffusion residual — same equation, physics-first weights."""
    du_dt  = dde.grad.jacobian(u, x, i=0, j=1)
    du_dx  = dde.grad.jacobian(u, x, i=0, j=0)
    du_dxx = dde.grad.hessian(u, x, i=0, j=0)
    return du_dt + V * du_dx - D * du_dxx


def boundary_conditions(geomtime):
    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.sin(np.pi * x[:, 0:1]),
        lambda _, on_initial: on_initial,
    )
    bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: np.zeros((len(x), 1)),
        lambda _, on_boundary: on_boundary,
    )
    return [bc, ic]


# ── Key change: expose loss_weights for the API to pick up ───────────────────
# The api.py build_model() currently hardcodes [1, 100, 100].
# Override by training via the standalone block below, or patch model.compile()
# in backend/model.py to read this value if present.
LOSS_WEIGHTS = [100, 10, 10]   # [PDE, BC, IC]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    geom       = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

    data = dde.data.TimePDE(
        geomtime, pde, boundary_conditions(geomtime),
        num_domain=3000, num_boundary=300, num_initial=300, num_test=2000,
    )
    net   = dde.nn.FNN([2] + [64] * 6 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    print(f"Loss weights: {LOSS_WEIGHTS}  (PDE-dominant)")
    model.compile("adam", lr=1e-3, loss_weights=LOSS_WEIGHTS)
    model.train(iterations=15000, display_every=1000)

    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # Quick error check
    x_pts = np.linspace(0, 1, 100)
    t_pts = np.full(100, 0.5)
    X     = np.column_stack([x_pts, t_pts])
    u     = model.predict(X).ravel()
    u_ex  = np.sin(np.pi * (x_pts - V * 0.5)) * np.exp(-D * np.pi**2 * 0.5)
    print(f"\nL2 error at t=0.5: {np.linalg.norm(u - u_ex) / np.linalg.norm(u_ex):.4f}")

    model.save("backend/weights/experiment_1_pde_dominant")
    print("Saved → backend/weights/experiment_1_pde_dominant-0.pt")
