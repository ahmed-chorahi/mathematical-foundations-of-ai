# Convexity

## Why It Matters for AI

Without convexity, gradient descent gives you *a* minimum and no promises. With convexity, every local minimum is global — the algorithm's output is provably the best possible. Convexity separates the models we fully understand (linear and logistic regression, SVMs) from the ones we train heuristically (deep networks). This chapter defines the property, gives the key theorem, and shows how to check it.

## Definition: Convex Function

A function $f$ is **convex** on a set $C$ if for all $\mathbf{x}, \mathbf{y} \in C$ and all $t \in [0,1]$:

$$
\boxed{\;f(t\mathbf{x} + (1-t)\mathbf{y}) \le t f(\mathbf{x}) + (1-t) f(\mathbf{y})\;}
$$

**Geometry:** the line segment connecting two points on the graph lies *above* the graph — the function bowls upward. Equivalently: its epigraph (region above the graph) is a convex set.

**Strict** convexity: strict inequality for $t \in (0,1)$ — the bowl has a single bottom (unique minimum).

A set $C$ is **convex** if every segment between its points stays in $C$: line segments, balls, half-spaces are convex; the ring shape ("annulus") is not (a chord can cross the hole).

## Examples

| function (on $\mathbb{R}$ or $\mathbb{R}^d$) | convex? | check |
|---|---|---|
| $x^2$, $\|x\|$, $e^x$, $|x|^p$ ($p \ge 1$) | yes | bowls upward |
| affine $ax + b$ | yes (and concave) | line: equality holds |
| $-x^2$, $-e^x$ | no (concave) | bowls downward |
| $x^3$ on $\mathbb{R}$ | **no** | concave for $x<0$, convex for $x>0$ — fails globally |
| $\ln x$ on $(0,\infty)$ | no (concave) | standard concave function |

**Sums and non-negative combinations preserve convexity:** if $f, g$ convex then $f + g$ convex, and $\lambda f$ convex for $\lambda \ge 0$. This is how compound objectives are verified: squared error is convex in the prediction; composing with an affine prediction keeps it convex in the parameters.

## The Key Theorem

> For a **convex** function on a convex set: **every local minimum is a global minimum**, and the set of minimizers is itself convex.

*Why* (intuition, not a proof): if a point were only locally optimal, some other point $\mathbf{y}$ has $f(\mathbf{y}) < f(\mathbf{x})$; the segment from $\mathbf{x}$ to $\mathbf{y}$ lies above $f$ along its length, and continuity means nearby points on the segment already have values below $f(\mathbf{x})$ — contradicting local optimality.

Consequences:

- Gradient descent's convergence point (a critical point of a convex problem) **is** the global optimum.
- **Strictly convex** ⇒ the optimum is unique — the model's parameters are determined by the objective.
- No saddle points with descent directions (a strictly convex function has none); no spurious local minima to escape.

**Non-convex case (deep networks):** many local minima and saddles exist. Practice: with enough width, most local minima have nearly equal loss (so "any minimum is fine"), and noise escapes saddles (section 04). These are empirical/research-level observations — the honest statement is that deep learning has *no global convergence guarantee*.

## Checking Convexity: Derivatives

**One dimension.** Twice-differentiable $f$ is convex on an interval iff

$$
f''(x) \ge 0 \quad \text{everywhere on the interval}
$$

($f'' > 0$ everywhere ⇒ strictly convex). Example: $(x^2)'' = 2 \ge 0$ ✓; $(x^3)'' = 6x$ — negative for $x < 0$ ✗.

**Several dimensions.** $f$ convex iff its Hessian is **positive semi-definite everywhere** on the domain:

$$
H_f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}
$$

(section 01: PSD means $\Delta^\top H \Delta \ge 0$ for all steps — no downward-curving direction anywhere). Positive definite everywhere ⇒ strictly convex.

**Worked example.** $f(x, y) = x^2 + 3xy + y^3$? Check the domain-conditional result from section 02: $H = \begin{bmatrix}2&3\\3&6y\end{bmatrix}$, determinant $12y - 9$, which is negative for $y < 3/4$ — so $H$ is not PSD everywhere; $f$ is **not** convex on $\mathbb{R}^2$ (consistent with its saddle point found in section 02).

**Worked example — linear regression is convex.** $J(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|^2 = (X\mathbf{w}-\mathbf{y})^\top(X\mathbf{w}-\mathbf{y})$. Expanding: $J = \mathbf{w}^\top X^\top X \mathbf{w} - 2\mathbf{y}^\top X \mathbf{w} + \mathbf{y}^\top\mathbf{y}$. Gradient: $\nabla J = 2X^\top(X\mathbf{w} - \mathbf{y})$; Hessian: $H = 2X^\top X$. And $X^\top X$ is always positive **semi**definite (section 01: $\mathbf{v}^\top X^\top X \mathbf{v} = \|X\mathbf{v}\|^2 \ge 0$) ⇒ **$J$ is convex**. If the columns of $X$ are independent, $X^\top X$ is positive definite ⇒ strictly convex ⇒ the unique normal-equations solution is the unique global minimizer. Everything from section 01 lines up: rank, projection, and convexity all describe the same well-posed problem.

## Composition Rules (Useful Subset)

- **Convex composed with affine is convex:** if $g$ is convex and $z(\theta) = A\theta + b$ is affine, then $g(z(\theta))$ is convex in $\theta$. This is the workhorse for linear models — any convex loss applied to $z = w^\top x + b$ is convex in $w$.
- **Pointwise maximum** of convex functions is convex (hinge-type losses).
- **Logistic loss check.** Binary cross-entropy for one logit is $\ell(z) = \ln(1 + e^{-z})$ (softplus of $-z$):
  $$
  \ell'(z) = -\sigma(-z), \qquad \ell''(z) = \sigma(-z)\big(1 - \sigma(-z)\big) > 0
  $$
  using the sigmoid derivative from section 02 — convex in $z$, hence (by the affine rule) convex in the weights. Logistic regression's objective is convex.

Deep networks: composing layers with non-linear activations leaves the convex class almost immediately — which is exactly why they are non-convex.

## Connection to AI

- **Linear regression** (normal equations) and **logistic regression** objectives are convex — unique optimum, reliable convergence, theory that says gradient descent finds the best model (section 07 uses this).
- **SVMs** are convex quadratic programs (with constraints — chapter 06) — solvable to proven optimality.
- **Non-convexity of deep learning** is the field's central theoretical gap: architecture and optimization choices are guided by practice and partial theory, not by convergence theorems.
- **Loss landscape research** (mode connectivity, sharpness analysis, the observation that wide networks have benign landscapes) is mathematics attempting to say something precise about the non-convex case.
- **Diagnosing training problems:** a convex objective that fails to converge indicates implementation or learning-rate bugs; a non-convex one failing may be landscape — different causes, different fixes.

## Common Mistakes

- Assuming "has a minimum" = convex — many non-convex functions have minima.
- Checking convexity only on a slice (along one direction) — the definition requires *all* segments; Hessian PSD everywhere is the practical test.
- Forgetting the domain: convexity claims are statements *on a stated domain* — $\ln x$ is concave on $(0,\infty)$ while $-\ln x$ is convex there, and a function convex on one region may fail on another ($x^3$).
- Believing convexity guarantees *fast* convergence — convexity guarantees *correctness* of the answer, not speed (rate depends on conditioning).
- Assuming sums of arbitrary functions stay convex — true for convex summands only.

## Exercises

1. Use the second-derivative test to classify $f(x) = x^2$, $g(x) = -\ln x$ (on $x > 0$), $h(x) = x^3$ (on all of $\mathbb{R}$) as convex/concave/neither.
2. Show $f(x) = e^x$ is convex via $f''(x) = e^x > 0$.
3. Verify the convexity inequality for $f(x) = x^2$ with $\mathbf{x} = 0$, $\mathbf{y} = 2$, $t = 1/2$ explicitly (both sides).
4. Compute the Hessian of $J(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|^2$ and show it is PSD for any $X$.
5. Is $f(x, y) = xy$ convex on $\mathbb{R}^2$? Check the Hessian ($f_{xx} = 0$, mixed $1$, $f_{yy} = 0$) and conclude via the $2 \times 2$ criterion.
6. State the practical difference in what you can conclude when training a convex model versus a deep network, in two sentences.
