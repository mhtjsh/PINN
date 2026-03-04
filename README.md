# PINNs

# Burgers Equation Using PINNs

## Problem Setup

The **1D viscous Burgers equation** is solved using a **Physics-Informed Neural Network (PINN)**:

$$
u_t + u u_x = \nu u_{xx}
$$

where

$$
\nu = \frac{0.01}{\pi}
$$

The spatial and temporal domains are:

- $x \in [-1,1]$
- $t \in [0,1]$

---

## Initial Condition

$$
u(x,0) = -\sin(\pi x)
$$

---

## Boundary Conditions

$$
u(-1,t) = 0
$$

$$
u(1,t) = 0
$$

This configuration is commonly used in **PINN benchmarks** because the solution develops **steep gradients during evolution**, making it a challenging nonlinear PDE problem.

---

# Standard PINN

## Model

A **fully connected neural network** with **tanh activation functions** is used to approximate the solution

$$
u(x,t)
$$

The PINN is trained by minimizing three components:

- **PDE residual loss**
- **Initial condition loss**
- **Boundary condition loss**

### Collocation Points

- **Interior collocation points** enforce the PDE.
- **Initial points** enforce the initial condition.
- **Boundary points** enforce the boundary conditions.

---

## Training Strategy

Training uses a **two-stage optimization scheme**:

1. **Adam optimizer** – coarse optimization  
2. **L-BFGS optimizer** – refinement

This hybrid strategy improves convergence when minimizing the physics residual.

---

## Results

The **standard PINN** reproduces the global structure of the Burgers solution:

- Sinusoidal initial condition
- Wave steepening over time
- Symmetric Burgers dynamics

The PDE residual remains small across most of the domain, with slightly higher values near the **shock region around $x \approx 0$**.

This region contains **steep gradients**, which are challenging for standard PINNs.

---

# Fourier Feature PINN

## Intended Hypotheses

Standard neural networks tend to learn **low-frequency components** more easily than **high-frequency structures**.

Since Burgers dynamics produce **sharp spatial gradients**, this can limit the model's representation capacity.

---

## Method

To address this limitation, **Fourier feature embeddings** are introduced before the neural network.

The input coordinates are mapped as

$$
(x,t) \rightarrow [\sin(Bx), \cos(Bx)]
$$

This expands the input representation using **periodic basis functions**.

---

## Results

The **Fourier PINN** produces:

- Sharper representation of the evolving wave
- Better modeling of **steep gradients near the shock region**

The **PDE residual heatmap** also shows a more consistent distribution of error across the domain.

---

# Diagnostics

Two visualizations are used to evaluate the model.

## Solution Heatmap

Shows the predicted Burgers solution across **space and time**.

## PDE Residual Heatmap

Displays the magnitude of the physics residual:

$$
\left| u_t + u u_x - \nu u_{xx} \right|
$$

This highlights regions where the **physics constraint is hardest for the network to satisfy**.

Residual values are lowest across most of the domain and increase slightly near the **shock region**, which is expected due to the large gradients present there.

---

# Summary

Both models successfully solve the **Burgers equation using physics-informed training**.

### Standard PINN
- Captures the overall dynamics
- Correctly models the global solution

### Fourier PINN
- Improves representation of **sharp spatial features**
- Better handles **high-frequency gradients**

These results demonstrate how **feature embeddings can enhance PINN performance when modeling nonlinear PDEs with steep gradients**.
