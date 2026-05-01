![EquilibriumDynamics](./equilibrium_dynamics_banner.svg)

# EquilibriumDynamics

> An interactive explainer covering the three pillars of modern neural differential equation solvers: the Infinite Time Neural ODE paradox, Physics-Informed Neural Network (PINN) training pipelines, and the hybrid NODE-ONet architecture blueprint.

---

## Live Demo

**[→ Open Interactive Explainer]
> **Note:** For full interactivity (sliders, training simulator, clickable architecture), download and open locally, or host via GitHub Pages.

---

## Overview

Classical numerical solvers integrate ODEs step-by-step from an initial condition to a target time T — a process whose cost scales as O(T). *EquilibriumDynamics* demonstrates, through interactive visualisation and annotated code, that this bottleneck can be broken in three complementary ways: by targeting steady states instead of fixed time-points, by encoding physics directly into the loss function, and by fusing both paradigms into a production-grade operator network.

The explainer is self-contained HTML/JS — no install, no server, no dependencies beyond a modern browser.

---

## Table of Contents

1. [Pillar 1 — Infinite Time Neural ODE](#pillar-1--infinite-time-neural-ode)
2. [Pillar 2 — PINN Training Pipeline](#pillar-2--pinn-training-pipeline)
3. [Pillar 3 — NODE-ONet Hybrid Architecture](#pillar-3--node-onet-hybrid-architecture)
4. [Interactive Explainer Features](#interactive-explainer-features)
5. [Code Examples](#code-examples)
6. [Method Comparison](#method-comparison)
7. [References](#references)
8. [Citation](#citation)

---

## Pillar 1 — Infinite Time Neural ODE

### The Paradox

Integrating a Neural ODE to a large fixed horizon T is computationally expensive: the adjoint method must reverse-integrate the same trajectory, accumulating function evaluations (NFE) proportionally to T and risking vanishing/exploding gradients along the way.

The Infinite Time Neural ODE resolves this by reframing the problem entirely. Instead of asking *"what is h(T)?"*, it asks *"where does the dynamics vanish?"*

```
Find h* such that  f(h*, θ) = 0
```

This fixed-point lives in state space — independent of time — and can be located with O(1) temporal cost.

### The Three Acceleration Techniques

#### 1. Steady-State Formulation
Define the loss on the equilibrium point h\* rather than on a trajectory endpoint. Systems with stable attractors (dissipative ODEs, implicit neural networks, energy-based models) all support this reformulation.

#### 2. Root-Finding Forward Pass
Replace the ODE solver with Broyden's quasi-Newton method:

```
h*_{n+1} = h*_n  −  J⁻¹ · f(h*_n, θ)
```

Broyden's method typically converges in 8–25 iterations regardless of the equivalent integration horizon T, reducing NFE by 10–100× on long-horizon problems.

#### 3. Implicit Differentiation (No Adjoint Integration)
Gradients are computed via the Implicit Function Theorem — a single linear solve rather than a full backward integration:

```
∂h*/∂θ  =  −(∂f/∂h*)⁻¹ · ∂f/∂θ
```

The Jacobian (∂f/∂h\*) is inverted once using conjugate gradient or LU factorisation. Backpropagation never unrolls through solver internals, completely eliminating the vanishing gradient pathology.

### Gauß–Legendre Quadrature (Non-Steady Systems)

When a true steady state does not exist, the reverse-time adjoint integral can be approximated in **parallel** using Gaussian quadrature nodes, evaluating multiple time-points simultaneously rather than sequentially:

| Solver | Typical NFE | Parallelisable? |
|---|---|---|
| Dormand–Prince RK45 | 100–300 | No |
| Gauß–Legendre (5-pt) | 5–15 | Yes |

> Reference: Norcliffe & Deisenroth, *Faster Training of Neural ODEs Using Gauß–Legendre Quadrature* (OpenReview).

---

## Pillar 2 — PINN Training Pipeline

### Core Concept

A Physics-Informed Neural Network approximates the solution y(x) of a differential equation by minimising a loss that encodes the **equation residual** — no labelled (x, y) pairs are required. The ODE itself provides supervision everywhere in the domain via automatic differentiation.

### Anatomy of the Loss Function

```
L_total  =  L_pde  +  L_ic
```

| Component | Expression | Role |
|---|---|---|
| Physics loss | `E_x[ (dy/dx + 2xy)² ]` | Enforces ODE satisfaction at collocation points |
| Initial condition loss | `(ŷ(0) − 1)²` | Anchors the solution to the known boundary value |

`dy/dx` is computed by `torch.autograd.grad` — exact, not finite-difference.

### Minimal PyTorch Implementation

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=2, width=50):
        super().__init__()
        net = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(width, width), nn.Tanh()]
        net += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


def physics_loss(model, x):
    x = x.requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    residual = dy_dx + 2 * x * y          # ODE: dy/dx + 2xy = 0
    return residual.pow(2).mean()


model   = PINN(layers=2, width=50)
optim   = torch.optim.Adam(model.parameters(), lr=1e-3)
x_col   = torch.linspace(0, 2, 100).view(-1, 1)
x0      = torch.tensor([[0.0]])

for epoch in range(2000):
    optim.zero_grad()
    loss = physics_loss(model, x_col) + (model(x0) - 1.0).pow(2)
    loss.backward()
    optim.step()
```

Exact solution for reference: `y(x) = exp(−x²)`

### Extended Examples

#### Burgers Equation (Nonlinear PDE, Shock Formation)
```python
# u: R² → R,  inputs: (x, t)
# PDE: ∂u/∂t + u·∂u/∂x  =  ν·∂²u/∂x²
u_t  = grad(u, t)
u_x  = grad(u, x)
u_xx = grad(u_x, x)
residual = u_t + u * u_x - nu * u_xx
```

#### Lorenz System (Chaotic ODE, Three Coupled Equations)
```python
# Three output heads: x_out, y_out, z_out
x_dot = grad(x_out, t);  y_dot = grad(y_out, t);  z_dot = grad(z_out, t)

residual = (
    (x_dot - sigma * (y_out - x_out))**2 +
    (y_dot - x_out * (rho - z_out) + y_out)**2 +
    (z_dot - x_out * y_out + beta * z_out)**2
)
```

> For chaotic systems, PINNs are reliable over short horizons. Long-horizon stability requires the Lyapunov-regularised NODE-ONet described in Pillar 3.

### Key Training Refinements

- **Adaptive collocation sampling** — concentrate training points in high-residual regions (residual-guided refinement, analogous to adaptive mesh refinement in FEM).
- **NTK-based loss weighting** — prevents the IC loss from dominating early training and starving the physics loss of gradient signal.
- **Curriculum learning** — gradually expand the spatial/temporal domain during training to improve convergence on stiff or oscillatory problems.

---

## Pillar 3 — NODE-ONet Hybrid Architecture

### Architecture Overview

NODE-ONet (Neural ODE Operator Network) fuses four specialised, independently-swappable components:

```
Sensor data
    │
    ▼
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
│ Encoder │ ──▶ │  Neural ODE  │ ──▶ │ Physics Head │ ──▶ │ Decoder │
│ CNN/MLP │     │  ż = f(z,θ)  │     │  PDE residual│     │z(t)→field│
│ high→z  │     │ latent evol. │     │  regularises │     │spatial  │
└─────────┘     └──────────────┘     └──────────────┘     └─────────┘
                                                                │
                                                                ▼
                                                       Solution field u(x,t)
```

| Component | Responsibility | Technical Implementation |
|---|---|---|
| Encoder | Dimensionality reduction | CNN or MLP mapping high-dim sensor data → latent z |
| Neural ODE layer | Temporal evolution | Solves ż = f(z, θ) in latent space |
| Physics head | Constraint enforcement | PDE residual gradient regularises θ |
| Decoder | Field reconstruction | Maps latent z(t) back to the physical spatial domain |

### Why Latent-Space Integration Matters

If your physical field lives on a 256×256 grid (65,536 dimensions), the encoder compresses it to a 64-dimensional latent vector z. The Neural ODE integrates z — reducing the ODE dimension by ~1,000× — making root-finding and quadrature methods tractable at scale.

### Three Key Advantages

#### Mesh-Free Operation
Operates on scattered sensor readings without requiring a structured 3D grid. Works with irregular, sparse, or moving observation points — directly applicable to real-world experimental data.

#### Operator Learning (Offline Training / Online Inference)
Learns the solution **operator** (initial condition → full trajectory), not a single solution instance. After offline training, the model predicts solutions for new initial conditions in a single forward pass — no re-solving required at deployment.

> This is implemented via a DeepONet-style branch/trunk split: the branch network encodes the IC, the trunk network encodes query coordinates (x, t), and their dot product yields u(x, t) for any queried point.

#### Lyapunov Exponent Regularisation
Finite-time Lyapunov Exponent (FTLE) regularisation monitors the learned vector field for chaotic divergence during training:

```python
# Penalise positive maximal Lyapunov exponent in latent ODE
ftle_loss = torch.relu(max_lyapunov_exponent(z_trajectory, f_theta))
L_total = L_data + lambda_pde * L_pde + lambda_ftle * ftle_loss
```

This prevents exponential error amplification in long-horizon predictions without constraining the model to artificially simple dynamics.

### Training vs Inference Workflow

**Offline training:**
1. Encode sensor data → z₀
2. Evolve z via Neural ODE (with adaptive solver or Gauß–Legendre quadrature)
3. Physics head evaluates PDE residual at decoded points
4. Backpropagate through all four components jointly

**Online inference (deployment):**
1. Encode new initial condition → z₀
2. Single forward pass through ODE layer
3. Decode → full solution field

No solver calls at inference time. Latency is sub-millisecond, replacing CFD/FEM solvers that may take hours per run.

---

## Interactive Explainer Features

The HTML explainer (`infinite_time_neural_ode_explainer.html`) provides four interactive tabs:

| Tab | Content | Interactive Elements |
|---|---|---|
| **Infinite Time Neural ODE** | Paradox explanation, three acceleration techniques, mathematical formulations | Clickable deep-dive buttons |
| **PINN Training** | Loss anatomy, training simulator | Layer/LR sliders, live loss bars, epoch counter |
| **Hybrid Architecture** | NODE-ONet component pipeline, training vs inference workflow | Clickable component cards |
| **Method Comparison** | Side-by-side scaling analysis | Problem-scale slider, live metric updates |

Open `infinite_time_neural_ode_explainer.html` directly in any modern browser — no server or installation required.

---

## Method Comparison

| Method | Solver calls (scale 5) | Gradient stability | Best use case |
|---|---|---|---|
| Standard Neural ODE | ~170 | Poor at large T | Short horizons, time-varying dynamics |
| PINN only | N/A (mesh-free) | Good | Known PDE, sparse data, single solution |
| Infinite-Time Neural ODE | ~12 | Excellent | Equilibrium systems, implicit networks |
| NODE-ONet (hybrid) | ~14 (training) / 1 (inference) | Excellent | Multi-query inference, production deployment |

---

## References

| # | Citation |
|---|---|
| 1 | arXiv (2025). *Deep Neural ODE Operator Networks for PDEs.* [arXiv:2510.15651](https://arxiv.org/html/2510.15651v1) |
| 2 | arXiv (2026). *Tracking Finite-Time Lyapunov Exponents to Robustify Neural ODEs.* [arXiv:2602.09613](https://arxiv.org/pdf/2602.09613) |
| 3 | DSpace@MIT (n.d.). *On Efficient Training & Inference of Neural Differential Equations.* [MIT DSpace](https://dspace.mit.edu/handle/1721.1/151379) |
| 4 | Frontiers in AI (2026). *Implementing physics-informed neural networks with deep learning for differential equations.* [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1717117/full) |
| 5 | MathWorks (n.d.). *Solve ODE Using Physics-Informed Neural Network.* [MATLAB Docs](https://www.mathworks.com/help/deeplearning/ug/solve-odes-using-a-neural-network.html) |
| 6 | Norcliffe & Deisenroth (n.d.). *Faster Training of Neural ODEs Using Gauß–Legendre Quadrature.* [OpenReview](https://openreview.net/forum?id=f0FSDAy1bU) |

---

## Citation

If you use this explainer or the methodology summary in academic work, please cite as:

```bibtex
@misc{equilibriumdynamics2026,
  title        = {EquilibriumDynamics: An Interactive Explainer for Infinite Time Neural ODEs, PINNs, and Hybrid Architectures},
  year         = {2026},
  howpublished = {\url{https://github.com/your-username/your-repo}},
  note         = {Interactive explainer: infinite\_time\_neural\_ode\_explainer.html}
}
```

---

*Built for academic dissemination. All code examples are PyTorch-style pseudocode intended for clarity — production implementations should add input normalisation, learning rate scheduling, and domain-specific boundary condition handling.*
