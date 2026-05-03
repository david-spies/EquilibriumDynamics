"""
EquilibriumDynamics — Experiment 2: Burgers' Equation
======================================================
Equation:  ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

Key difference from convection-diffusion:
  The advection velocity is now u itself (nonlinear self-advection).
  This causes wave steepening and, for small ν, shock formation.

Physics significance:
  • Tests whether mass conservation holds when the velocity field is
    solution-dependent (unlike linear convection where v is constant).
  • The nonlinear term u·∂u/∂x creates energy cascade — a fundamentally
    harder problem for PINNs than linear transport.
  • At ν=0.01 (Re≈100), a near-shock forms around t=0.5 that challenges
    the smooth tanh network to represent a steep gradient.

IC:  u(x,0) = sin(π·x)   — same as baseline for direct comparison
BCs: u(0,t) = u(1,t) = 0 — same Dirichlet zero-flux boundaries

Drop this file into the Physics dropzone, then click Train PINN.
"""

import numpy as np
import deepxde as dde

NU = 0.01   # kinematic viscosity  (analogous to D in convection-diffusion)


def pde(x, u):
    """
    Burgers' equation residual:
        R = ∂u/∂t + u·∂u/∂x − ν·∂²u/∂x²

    Note: u·∂u/∂x is the nonlinear advection term.
    Automatic differentiation handles this exactly — no linearisation needed.
    """
    du_dt  = dde.grad.jacobian(u, x, i=0, j=1)   # ∂u/∂t
    du_dx  = dde.grad.jacobian(u, x, i=0, j=0)   # ∂u/∂x
    du_dxx = dde.grad.hessian(u, x, i=0, j=0)    # ∂²u/∂x²
    return du_dt + u * du_dx - NU * du_dxx        # nonlinear: u·∂u/∂x


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


# Higher PDE weight because the nonlinear shock region needs extra emphasis
LOSS_WEIGHTS = [100, 10, 10]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    print("=" * 60)
    print(f"  Burgers' Equation  |  ν = {NU}  |  Re ≈ {1/NU:.0f}")
    print("  Nonlinear shock formation expected near t=0.5")
    print("=" * 60)

    geom       = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

    # More domain points to resolve near-shock region
    data = dde.data.TimePDE(
        geomtime, pde, boundary_conditions(geomtime),
        num_domain=5000, num_boundary=400, num_initial=400, num_test=3000,
    )

    # Slightly wider network to capture sharp gradients
    net   = dde.nn.FNN([2] + [64] * 6 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=LOSS_WEIGHTS)
    model.train(iterations=20000, display_every=1000)

    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # Visualise solution field
    n_x, n_t = 100, 100
    XX, TT   = np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_t))
    XY       = np.column_stack([XX.ravel(), TT.ravel()])
    U        = model.predict(XY).reshape(n_t, n_x)

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(f"Burgers' Equation PINN  |  ν={NU}  |  Re≈{1/NU:.0f}", fontsize=13)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(XX, TT, U, cmap="plasma", alpha=0.9)
    ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.set_title("u(x,t) — PINN")
    ax1.view_init(elev=28, azim=-55)

    ax2 = fig.add_subplot(1, 2, 2)
    for t_snap in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(t_snap * (n_t - 1))
        ax2.plot(np.linspace(0, 1, n_x), U[idx], label=f"t={t_snap:.2f}")
    ax2.set_xlabel("x"); ax2.set_ylabel("u"); ax2.set_title("Snapshots")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Figure_Burgers.png", dpi=150, bbox_inches="tight")
    print("Figure_Burgers.png saved")

    model.save("backend/weights/experiment_2_burgers")
    print("Saved → backend/weights/experiment_2_burgers-0.pt")
