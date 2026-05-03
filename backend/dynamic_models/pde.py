"""
EquilibriumDynamics — Plug-and-Play PDE Definition
1D Convection-Diffusion: du/dt + v*du/dx - D*d²u/dx² = 0

Drop this file into the Physics dropzone in the UI.
The backend will hot-load `pde(x, u)` without restarting the server.

This file ALSO contains a standalone __main__ block that:
  1. Trains the full DeepXDE PINN (Adam + L-BFGS)
  2. Compares PINN vs analytical solution in a 3-panel surface plot
  3. Saves Figure_1.png to the project root

Run standalone:
    python backend/dynamic_models/pde.py
"""

import numpy as np
import deepxde as dde


# ─── Physical parameters ─────────────────────────────────────────────────────
V = 1.0    # convection velocity
D = 0.01   # diffusion coefficient


# ══════════════════════════════════════════════════════════════════════════════
# PDE residual — the function the API hot-loads.
# MUST be named exactly `pde(x, u)`.
# ══════════════════════════════════════════════════════════════════════════════
def pde(x, u):
    """
    Residual of the 1D convection-diffusion equation.
      x[:, 0] = spatial coordinate
      x[:, 1] = time coordinate
    """
    du_dt  = dde.grad.jacobian(u, x, i=0, j=1)
    du_dx  = dde.grad.jacobian(u, x, i=0, j=0)
    du_dxx = dde.grad.hessian(u, x, i=0, j=0)
    return du_dt + V * du_dx - D * du_dxx


# ══════════════════════════════════════════════════════════════════════════════
# Analytical solution — for comparison/error analysis only
# ══════════════════════════════════════════════════════════════════════════════
def analytical(x, t):
    """
    Exact solution for IC = sin(π·x), Dirichlet BCs u(0,t)=u(1,t)=0.

    For pure convection-diffusion on a periodic/infinite domain:
        u(x,t) = sin(π·(x − v·t)) · exp(−D·π²·t)

    The Dirichlet BCs at x=0,1 are enforced by the PINN loss, not the
    analytical formula. The analytical surface is shown for qualitative
    comparison of shape and amplitude.
    """
    return np.sin(np.pi * (x - V * t)) * np.exp(-D * np.pi**2 * t)


# ══════════════════════════════════════════════════════════════════════════════
# Optional: boundary_conditions() — used by the API orchestrator
# ══════════════════════════════════════════════════════════════════════════════
def boundary_conditions(geomtime):
    """Returns [bc, ic] compatible with dde.data.TimePDE."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Standalone — python backend/dynamic_models/pde.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # save to file without opening a window
    import matplotlib.pyplot as plt
    from matplotlib import cm

    print("=" * 60)
    print("  EquilibriumDynamics — Standalone PINN Training")
    print(f"  PDE: du/dt + {V}·du/dx − {D}·d²u/dx² = 0")
    print(f"  Péclet number Pe = v/D = {V/D:.0f}")
    print("=" * 60)

    # ── Domain ────────────────────────────────────────────────────────────
    geom       = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)
    conditions = boundary_conditions(geomtime)

    data = dde.data.TimePDE(
        geomtime, pde, conditions,
        num_domain=3000, num_boundary=300, num_initial=300,
        num_test=2000,
    )

    # ── Network: 6 × 64, tanh ────────────────────────────────────────────
    net   = dde.nn.FNN([2] + [64] * 6 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # ── Phase 1: Adam ─────────────────────────────────────────────────────
    print("\nPhase 1: Adam (10 000 iterations)...")
    model.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
    model.train(iterations=10000, display_every=1000)
    print("Adam complete.\n")

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────
    print("Phase 2: L-BFGS refinement...")
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    print("L-BFGS complete.\n")

    # ── Evaluation grid ───────────────────────────────────────────────────
    n_x, n_t = 80, 80
    x_vals   = np.linspace(0, 1, n_x)
    t_vals   = np.linspace(0, 1, n_t)
    XX, TT   = np.meshgrid(x_vals, t_vals)
    XY       = np.column_stack([XX.ravel(), TT.ravel()])

    u_pinn = model.predict(XY).reshape(n_t, n_x)
    u_anal = analytical(XX, TT)
    u_err  = np.abs(u_pinn - u_anal)

    print(f"Pointwise error — max: {u_err.max():.4f}  mean: {u_err.mean():.4f}")

    # ── Figure_1: three-panel comparison ──────────────────────────────────
    fig = plt.figure(figsize=(19, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"EquilibriumDynamics PINN  |  v={V}, D={D}, Pe={V/D:.0f}"
        f"  |  max‖error‖ = {u_err.max():.4f}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Shared scale for PINN and analytical panels
    vmin = min(u_pinn.min(), u_anal.min())
    vmax = max(u_pinn.max(), u_anal.max())

    panels = [
        (u_pinn, f"PINN Solution  $\\hat{{u}}(x,t)$",  vmin, vmax, "viridis"),
        (u_anal,  "Analytical Solution  $u(x,t)$",      vmin, vmax, "viridis"),
        (u_err,   "Pointwise Error  $|\\hat{u}-u|$",    0,    u_err.max(), "plasma"),
    ]

    for i, (Z, title, zlo, zhi, cmap) in enumerate(panels):
        ax   = fig.add_subplot(1, 3, i + 1, projection="3d")
        norm = plt.Normalize(vmin=zlo, vmax=zhi)
        ax.plot_surface(
            XX, TT, Z,
            facecolors=cm.get_cmap(cmap)(norm(Z)),
            linewidth=0, antialiased=True, alpha=0.93,
        )
        ax.set_xlabel("x", labelpad=5)
        ax.set_ylabel("t", labelpad=5)
        ax.set_zlabel("u", labelpad=5)
        ax.set_title(title, pad=10, fontsize=11)
        ax.view_init(elev=28, azim=-55)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.52, pad=0.08)

    plt.tight_layout()
    out = "Figure_1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure_1 saved → {out}")
