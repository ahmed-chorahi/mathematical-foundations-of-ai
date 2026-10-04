# Jacobian and Hessian

## Why It Matters for AI

Two derived objects organize everything calculus needs about a multivariable function: the **Jacobian** collects all first derivatives, the **Hessian** all second derivatives. The Jacobian is the matrix form of the gradient and generalizes differentiation to layers that output vectors; the Hessian describes curvature — sharp versus flat directions of the loss — and decides whether a zero-gradient point is a minimum, a maximum, or a saddle.

## Jacobian

For a vector-valued function $F: \mathbb{R}^n \to \mathbb{R}^m$, component $i$ being $F_i$:

$$
J_F = \begin{bmatrix}
\dfrac{\partial F_1}{\partial x_1} & \cdots & \dfrac{\partial F_1}{\partial x_n} \\[6pt]
\vdots & & \vdots \\[4pt]
\dfrac{\partial F_m}{\partial x_1} & \cdots & \dfrac{\partial F_m}{\partial x_n}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
$$

Row $i$ = gradient of output $i$; column $j$ = how output vector responds to input $j$.

**Shapes.** $m$ outputs, $n$ inputs → $m \times n$. For a scalar function ($m = 1$), the Jacobian is the $1 \times n$ row form of the gradient: $\nabla f$ as a column, $J_f$ as a row — same numbers.

**Worked example — scalar case.** $f(x,y) = x^2 + 3xy$ (chapter 03):

$$
J_f = \begin{bmatrix} 2x + 3y & 3x \end{bmatrix}
\quad (1 \times 2), \qquad
\nabla f = \begin{bmatrix} 2x + 3y \\ 3x \end{bmatrix} \quad (2 \times 1)
$$

**Worked example — vector case.** $F(u, v) = \begin{bmatrix} u + v \\ uv \end{bmatrix}$:

$$
J_F = \begin{bmatrix}
\partial(u+v)/\partial u & \partial(u+v)/\partial v \\
\partial(uv)/\partial u & \partial(uv)/\partial v
\end{bmatrix}
= \begin{bmatrix} 1 & 1 \\ v & u \end{bmatrix}
$$

At $(u,v) = (2,3)$: $J_F = \begin{bmatrix}1&1\\3&2\end{bmatrix}$ — e.g., entry $(2,1) = 3$ says the second output $uv$ changes by $3$ per unit increase in $u$ at that point.

**Linearization.** For small $\Delta\mathbf{x}$:

$$
F(\mathbf{x} + \Delta\mathbf{x}) \approx F(\mathbf{x}) + J_F\,\Delta\mathbf{x}
$$

The Jacobian is the best linear approximation of $F$ near $\mathbf{x}$ — the multivariable tangent, matrix-shaped. A neural-network layer with activation is non-linear globally but linear through its Jacobian at a point; that local view is what the backward pass manipulates (section 08).

## Hessian

Second derivatives of a scalar $f: \mathbb{R}^n \to \mathbb{R}$, arranged square:

$$
H_f = \begin{bmatrix}
\dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & & \ddots
\end{bmatrix}
\in \mathbb{R}^{n \times n}
$$

Notation: $\dfrac{\partial^2 f}{\partial x_i \partial x_j} = f_{x_i x_j}$ — first differentiate by $x_j$, then by $x_i$ (order written right-to-left in the denominator; mixed entries come in pairs).

**Symmetry.** For functions with continuous second partials (Schwarz's theorem):

$$
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}
\quad\Longrightarrow\quad
H_f = H_f^\top
$$

The Hessian is always symmetric — which is exactly the good case of section 01: real eigenvalues, orthogonal eigendecomposition.

**Worked example.** $f(x,y) = x^2 + 3xy + y^3$:

$$
f_x = 2x + 3y \;\Rightarrow\; f_{xx} = 2,\; f_{xy} = 3
$$
$$
f_y = 3x + 3y^2 \;\Rightarrow\; f_{yx} = 3,\; f_{yy} = 6y
$$

$$
H_f = \begin{bmatrix} 2 & 3 \\ 3 & 6y \end{bmatrix}
$$

Mixed entries match ($3 = 3$) ✓. At $(0,0)$: $H = \begin{bmatrix}2&3\\3&0\end{bmatrix}$.

**Reading the Hessian.** For a quadratic $f(\mathbf{x}) \approx \tfrac12 \Delta\mathbf{x}^\top H \Delta\mathbf{x}$ near a critical point:

| Hessian at the point | meaning |
|---|---|
| all eigenvalues $> 0$ (**positive definite**) | curves up in every direction → local minimum |
| all eigenvalues $< 0$ (**negative definite**) | curves down everywhere → local maximum |
| eigenvalues of both signs (**indefinite**) | up in some directions, down in others → **saddle point** |
| some eigenvalues $= 0$ | flat direction — test inconclusive |

This is the second-derivative test of section 07 in matrix language, and it uses section 01's eigenvalue theory directly: the eigenvalues are the curvatures along the principal (eigenvector) directions.

**Why saddle points matter.** In many parameters ($n$ large), a zero gradient with one negative eigenvalue is common: the point is optimal along $n-1$ directions but a downhill slope remains along one. High-dimensional losses have many saddles — which is why "the gradient is zero" rarely means "training is finished", and why gradient-based methods (which follow directions of negative curvature automatically when descending) outperform methods that demand strict minima.

## The Two Objects Side by Side

| | Jacobian | Hessian |
|---|---|---|
| applies to | any differentiable $F$ | scalar $f$ |
| entries | all first partials | all second partials |
| shape | $m \times n$ | $n \times n$ |
| role | local linear approximation | local curvature |
| symmetry | not symmetric in general | always symmetric |
| in AI | backpropagation's per-layer local maps | curvature, second-order optimization, convergence analysis |

## Connection to AI

- **Backpropagation is Jacobian algebra.** Each layer's local linearization is its Jacobian; the chain rule for vectors (chapter 05) multiplies Jacobians along the network. Section 08 works this out for a concrete network.
- **Per-parameter sensitivities.** $\partial \text{loss}/\partial \theta_i$ are the entries of the gradient; the Jacobian of the *predictions* w.r.t. parameters tells you which input directions a parameter controls.
- **Newton's method** uses $H^{-1}\nabla J$ to scale steps by inverse curvature — the mathematical origin of second-order optimizers (mentioned in section 05; the positive-definiteness requirement comes from this chapter).
- **Curvature and generalization.** Flat minima (small Hessian eigenvalues) are associated with better generalization in practice; sharp minima overfit. This is an empirical claim, not a theorem — but *stating* it requires exactly this vocabulary.
- **The Hessian of a neural network is enormous** ($d \times d$ parameters), so exact computation is rarely feasible; matrix-free methods and approximations (K-FAC, Hutchinson estimators) exist precisely because $H$ is the right object but too large — knowing what it is tells you what is being approximated.

## Common Mistakes

- Transposing $J_F$ or confusing row/column conventions — fix the shape first ($m \times n$: outputs × inputs), then fill entries.
- Mixing the order of mixed partials without symmetry justification — equality holds for *nice* functions; pathological counterexamples exist but never appear in ML (all ML functions are compositions of smooth pieces or piecewise smooth).
- Reading $f_{xy}$ as "$f_x$ times $y$" — it is a nested derivative.
- Assuming the Hessian's diagonal alone captures curvature — off-diagonal terms couple directions; a diagonal-only view misses rotation of principal directions.
- Equating $\nabla f = \mathbf{0}$ with a minimum without checking eigenvalues of $H$.

## Exercises

1. Write the Jacobian of $F(x, y) = \begin{bmatrix} x^2 \\ xy \\ y \end{bmatrix}$ and give its shape.
2. Compute the Jacobian of $f(x,y) = x^2 + 3xy + y^3$ at $(1, 1)$ in both row (Jacobian) and column (gradient) form.
3. For $F(u,v) = (u+v, uv)$, evaluate $J_F$ at $(u,v) = (2,3)$ and explain in words the entry $J_{21}$.
4. Compute the Hessian of $f(x,y) = x^2 + 3xy + y^3$ at $(0,0)$ and at $(0,1)$; is it symmetric in both cases?
5. The Hessian of a loss at a critical point has eigenvalues $2, 5, -1$. What type of critical point is it?
6. Show directly that the mixed partials of $f(x,y) = xy^2$ agree: compute $\partial^2 f/\partial x \partial y$ and $\partial^2 f/\partial y \partial x$.
