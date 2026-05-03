"""
EquilibriumDynamics — Experiment 3: Reaction-Diffusion with Source/Sink
========================================================================
Equation:  ∂u/∂t + v·∂u/∂x − D·∂²u/∂x² = k·u·(1 − u)

This adds a logistic reaction term R(u) = k·u·(1−u) to the baseline PDE.
This is the Fisher-KPP (Kolmogorov-Petrovsky-Piskunov) equation — one of
the most studied PDEs in mathematical biology and combustion theory.

Physics significance:
  • The reaction term creates a travelling wave front that propagates at a
    well-defined speed c = 2√(kD), independently of convection.
  • Mass is no longer conserved globally (the source term creates mass).
    This tests whether the PINN can distinguish "conservation" from
    "steady growth" — a fundamental test of physical reasoning.
  • Combines all three phenomena: convection, diffusion, reaction.

IC:  u(x,0) = 0.5·sin(π·x)   — half amplitude to avoid saturation at t=0
BCs: u(0,t) = u(1,t) = 0

Curriculum learning strategy (implemented below):
  Phase 1 (Adam 0→5k):    D=0.1  (diffusion-dominated, smooth, easy)
  Phase 2 (Adam 5k→15k):  D=0.01 (convection-diffusion regime)
  Phase 3 (L-BFGS):       Full nonlinear reaction term

Drop this file into the Physics dropzone, then click Train PINN.
"""

import numpy as np
import deepxde as dde

V  = 1.0    # convection
D  = 0.01   # diffusion
K  = 1.0    # reaction rate  (logistic growth)


def pde(x, u):
    """
    Fisher-KPP residual:
        R = ∂u/∂t + v·∂u/∂x − D·∂²u/∂x² − k·u·(1−u)

    The reaction term k·u·(1−u) is zero at u=0 and u=1 (fixed points)
    and maximum at u=0.5.
    """
    du_dt  = dde.grad.jacobian(u, x, i=0, j=1)
    du_dx  = dde.grad.jacobian(u, x, i=0, j=0)
    du_dxx = dde.grad.hessian(u, x, i=0, j=0)
    reaction = K * u * (1.0 - u)
    return du_dt + V * du_dx - D * du_dxx - reaction


def pde_easy(x, u):
    """
    Curriculum Phase 1: high diffusion (D=0.1), no reaction.
    The network learns the basic shape before tackling sharp gradients.
    """
    du_dt  = dde.grad.jacobian(u, x, i=0, j=1)
    du_dx  = dde.grad.jacobian(u, x, i=0, j=0)
    du_dxx = dde.grad.hessian(u, x, i=0, j=0)
    return du_dt + V * du_dx - 0.1 * du_dxx   # D=0.1, no reaction


def boundary_conditions(geomtime):
    ic = dde.icbc.IC(
        geomtime,
        lambda x: 0.5 * np.sin(np.pi * x[:, 0:1]),   # half-amplitude IC
        lambda _, on_initial: on_initial,
    )
    bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: np.zeros((len(x), 1)),
        lambda _, on_boundary: on_boundary,
    )
    return [bc, ic]


LOSS_WEIGHTS = [100, 10, 10]   # PDE-dominant (physics first)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print(f"  Fisher-KPP Equation  |  v={V}, D={D}, k={K}")
    print(f"  Travelling wave speed c = 2√(kD) = {2*(K*D)**0.5:.4f}")
    print("=" * 60)

    geom       = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)
    conditions = boundary_conditions(geomtime)

    data = dde.data.TimePDE(
        geomtime, pde_easy, conditions,
        num_domain=4000, num_boundary=400, num_initial=400, num_test=2000,
    )
    net   = dde.nn.FNN([2] + [64] * 6 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # ── Curriculum Phase 1: easy (high diffusion, no reaction) ────────────
    print("\nCurriculum Phase 1: high diffusion D=0.1, no reaction (5000 iters)")
    model.compile("adam", lr=1e-3, loss_weights=LOSS_WEIGHTS)
    model.train(iterations=5000, display_every=1000)

    # ── Curriculum Phase 2: swap in full PDE ──────────────────────────────
    print("\nCurriculum Phase 2: full Fisher-KPP PDE (10000 iters)")
    model.data.pde = pde   # hot-swap the PDE residual function
    model.compile("adam", lr=5e-4, loss_weights=LOSS_WEIGHTS)
    model.train(iterations=10000, display_every=1000)

    # ── Phase 3: L-BFGS polish ────────────────────────────────────────────
    print("\nPhase 3: L-BFGS refinement")
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # ── Visualise ─────────────────────────────────────────────────────────
    n_x, n_t = 100, 100
    XX, TT   = np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_t))
    U        = model.predict(np.column_stack([XX.ravel(), TT.ravel()])).reshape(n_t, n_x)

    # Mass over time (should INCREASE due to source term)
    masses = [np.trapezoid(U[i], np.linspace(0, 1, n_x)) for i in range(n_t)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Fisher-KPP  |  v={V}, D={D}, k={K}", fontsize=13)

    axes[0].contourf(np.linspace(0,1,n_x), np.linspace(0,1,n_t), U, levels=30, cmap="viridis")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t"); axes[0].set_title("u(x,t) heatmap")

    axes[1].plot(np.linspace(0, 1, n_t), masses, color="teal", lw=2)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("∫u dx")
    axes[1].set_title("Mass over time (source term → growth)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Figure_FisherKPP.png", dpi=150, bbox_inches="tight")
    print("Figure_FisherKPP.png saved")

    model.save("backend/weights/experiment_3_fisher_kpp")
    print("Saved → backend/weights/experiment_3_fisher_kpp-0.pt")
