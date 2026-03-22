# Physics-Informed Neural Networks (PINNs)

This repository contains experiments with **Physics-Informed Neural Networks (PINNs)** for solving partial differential equations by incorporating governing physics directly into the training objective.

Currently, the repository includes:

- Burgers Equation (Standard vs Fourier PINN)  
- Electrostatic Potential via Laplace Equation  

---

## 1. Burgers Equation

Link: https://github.com/mhtjsh/PINN/blob/Primary/Burges%20Equation%20Standard%20vs%20Fourier%20Training%20PINN.ipynb

### Problem

We solve the 1D viscous Burgers equation:

$$
u_t + u u_x = \nu u_{xx}, \quad \nu = \frac{0.01}{\pi}
$$

**Domain:**

- $x \in [-1,1]$  
- $t \in [0,1]$  

**Initial condition:**

$$
u(x,0) = -\sin(\pi x)
$$

**Boundary conditions:**

$$
u(-1,t) = 0, \quad u(1,t) = 0
$$

This setup is a standard PINN benchmark due to the formation of **steep gradients (shock-like behavior)**.

---

### Approach

A fully connected neural network approximates $u(x,t)$ and is trained using:

- PDE residual loss  
- Initial condition loss  
- Boundary condition loss  

**Residual:**

$$
f(x,t) = u_t + u u_x - \nu u_{xx}
$$

**Training strategy:**

- Adam (initial convergence)  
- L-BFGS (refinement)  

---

### Fourier Feature Variant

To mitigate spectral bias, inputs are mapped using Fourier features:

$$
(x,t) \rightarrow [\sin(Bx), \cos(Bx)]
$$

This improves the model’s ability to represent **high-frequency components**, particularly near the shock region.

---

### Observations

- The **standard PINN** captures global dynamics but shows higher residuals near $x \approx 0$.  
- The **Fourier PINN** improves resolution of steep gradients and yields more uniform residuals.  

---

## 2. Electrostatic Potential (Laplace Equation)

Link: https://github.com/mhtjsh/PINN/blob/Primary/PINN_w_electorstatic_potential_through_Laplace_eqn.ipynb

### Problem

We solve the 2D Laplace equation:

$$
\nabla^2 \phi = 0
$$

or equivalently,

$$
\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} = 0
$$

This governs electrostatic potential in regions with no charge.

---

### Approach

A neural network approximates $\phi(x,y)$.

The loss consists of:

- **PDE residual:**

$$
f(x,y) = \phi_{xx} + \phi_{yy}
$$

- **Boundary condition loss**

$$
\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{BC}
$$

Second-order derivatives are computed using automatic differentiation.

---

### Observations

- The model learns smooth potential fields consistent with Laplace dynamics.  
- Results are strongly influenced by boundary conditions, as expected for elliptic PDEs.  
- Compared to Burgers, convergence is more stable due to the absence of nonlinear advection.  

---

## Summary

- **Burgers Equation** highlights challenges in PINNs for nonlinear PDEs with sharp gradients and demonstrates the benefit of Fourier feature embeddings.  
- **Laplace Equation** provides a stable setting illustrating PINNs for elliptic PDEs governed primarily by boundary conditions.  

Overall, these notebooks illustrate both the **strengths and limitations of PINNs**, particularly the role of representation and optimization in solving different classes of PDEs.
