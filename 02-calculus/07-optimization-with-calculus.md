# Critical Points and the Second-Derivative Test

## Why It Matters for AI

Training ends where the gradient vanishes — or does not vanish, and training ends anyway. Knowing what a zero gradient means (minimum? saddle? maximum?), how to classify it, and what can still go wrong is the calculus-level completion of the optimization picture. Section 05 builds the iterative methods; this chapter provides the theory they aim at.

## Critical Points

A **critical point** of $f$ is a point where $f'(\mathbf{x}) = 0$ (for several variables: $\nabla f = \mathbf{0}$, all partials vanish).

Why they matter: for a differentiable function on an open domain, a local extremum **must** be a critical point — at an interior maximum the tangent cannot tilt up or down (it would contradict maximality in both directions), so it is flat, $f' = 0$.

Caveat: extrema can also occur at **boundaries** of the domain (e.g., parameters constrained to $\theta \ge 0$) — there the derivative need not vanish. Unconstrained training problems are the interior case.

## Second-Derivative Test (One Variable)

At a critical point $a$ (so $f'(a) = 0$):

| condition | conclusion |
|---|---|
| $f''(a) > 0$ | local **minimum** |
| $f''(a) < 0$ | local **maximum** |
| $f''(a) = 0$ | test **inconclusive** — need more analysis |

*Why it works* (second-order Taylor from chapter 06): near $a$,

$$
f(a+h) \approx f(a) + \underbrace{f'(a)}_{=\,0}h + \tfrac12 f''(a)h^2
$$

The linear term vanishes at a critical point, so the sign of $f''$ decides the shape: positive → opens up → minimum; negative → opens down → maximum.

**Worked example — minimum.** $f(x) = x^2 - 4x + 5$:

$f'(x) = 2x - 4 = 0 \Rightarrow x = 2$. $f''(2) = 2 > 0$ → local minimum, $f(2) = 1$.

(Global? $f(x) = (x-2)^2 + 1 \ge 1$ everywhere → the local minimum is also global.)

**Worked example — inconclusive.** $f(x) = x^3$: $f'(0) = 0$ (critical), $f''(0) = 0$ — inconclusive. Check directly: $x^3$ increases through $0$; not an extremum, an **inflection point**. The test failing means "look closer", not "there is an extremum".

## Several Variables: the Hessian Test

At $\nabla f = \mathbf{0}$, examine the Hessian $H$ (chapter 04) at that point:

| eigenvalues of $H$ | conclusion |
|---|---|
| all $> 0$ (positive definite) | local minimum |
| all $< 0$ (negative definite) | local maximum |
| both positive and negative (indefinite) | **saddle point** — minimum along some directions, maximum along others |
| some zeros | inconclusive |

**Worked example — saddle.** $f(x, y) = x^2 - y^2$: $\nabla f = (2x, -2y) = \mathbf{0}$ at $(0,0)$. $H = \begin{bmatrix}2&0\\0&-2\end{bmatrix}$ — eigenvalues $2 > 0$ and $-2 < 0$: indefinite → saddle. Along $x$-axis it is a minimum ($x^2$), along $y$-axis a maximum ($-y^2$).

**Worked example — minimum in 2D.** $f(x,y) = x^2 + 3xy + y^3$ from chapter 04. Critical points solve $\nabla f = \mathbf{0}$:

$$
2x + 3y = 0, \qquad 3x + 3y^2 = 0
$$

From the first: $x = -\tfrac{3}{2}y$. Substitute: $3(-\tfrac32 y) + 3y^2 = 0 \Rightarrow 3y(y - \tfrac32) = 0 \Rightarrow y = 0$ or $y = \tfrac32$.

- $y = 0 \Rightarrow x = 0$: point $(0,0)$, $H = \begin{bmatrix}2&3\\3&0\end{bmatrix}$, $\det = -9 < 0$ → indefinite → **saddle**.
- $y = \tfrac32 \Rightarrow x = -\tfrac94$: $H = \begin{bmatrix}2&3\\3&9\end{bmatrix}$, $\det = 18 - 9 = 9 > 0$ and $f_{xx} = 2 > 0$ → positive definite → **local minimum**.

(The $2\times2$ shortcut: positive definite $\iff$ $f_{xx} > 0$ and $\det H > 0$ — both eigenvalue signs positive is equivalent to these two numbers for $2\times2$.)

## Local vs Global

- **Local minimum:** smallest value among nearby points.
- **Global minimum:** smallest value over the whole domain.

Every global minimum on an open domain is a local minimum (hence a critical point); the reverse fails. For **convex** functions — every line segment between two points lies above the graph — local = global, and the theory becomes clean (section 05 builds this properly). Neural-network losses are *not* convex: they have many local minima and saddles.

**Practical view.** In high dimensions, gradient descent rarely needs a *perfect* critical point: it wanders toward regions of low loss and stops when gradients are small relative to noise. The calculus describes the ideal; section 05 describes the practice.

## Flat Regions and Non-Differentiable Points

- **Plateaus:** regions where $\nabla f = \mathbf{0}$ over an area — progress stalls without an extremum. Saddle plateaus are common in deep networks.
- **ReLU's corner:** $f$ not differentiable at kinks; classical critical-point theory does not apply exactly there. Subgradient methods (using a valid slope at the kink, e.g., $0$) handle it; practical training does fine because kinks are sparse.

## Connection to AI

- **$\nabla J = \mathbf{0}$ is the ideal stopping condition** — but rarely the actual one. Training stops by patience, budget, or gradient noise, not by solving the equation exactly.
- **Saddles, not minima, dominate flat regions** of high-dimensional losses — hence the emphasis in ML on methods that escape saddles (noise from minibatches does this naturally; strict gradient descent can stall).
- **Positive definiteness of the Hessian** at a solution certifies a genuine minimum and underpins Newton's method (needs $H^{-1}$ to exist and point downhill).
- **Convexity = guaranteed global optimum** — the reason linear/logistic regression are fully understood, and why neural networks (non-convex) are not.

## Common Mistakes

- Concluding "extremum" from $f' = 0$ alone — $x^3$ at $0$ is the counterexample; always test.
- Forgetting boundary points when the domain is restricted.
- Checking only the Hessian's diagonal — off-diagonal entries can flip definiteness (as in the saddle example: diagonal entries $2, 0$ looked non-negative).
- Confusing "critical point" (defined: gradient zero) with "optimum" (requires testing).
- Assuming gradient descent finds the global minimum — that conclusion requires convexity (section 05), not calculus.

## Exercises

1. Find all critical points of $f(x) = x^2 - 6x + 11$ and classify them with the second-derivative test.
2. Classify the critical point of $f(x) = -x^2 + 4$.
3. Find and classify the critical points of $g(x) = x^3 - 3x$: expect two — one minimum, one maximum ($g'(x) = 3x^2 - 3$).
4. For $f(x,y) = x^2 + y^2$, find the critical point and classify it via the Hessian.
5. For $f(x,y) = x^2 - y^2$, verify $\nabla f = \mathbf{0}$ at the origin and classify.
6. Compute $\nabla f$ for $f(x,y) = (x-1)^2 + (y-2)^2$ and solve $\nabla f = \mathbf{0}$; explain why the minimum value $0$ must be global for this function.
