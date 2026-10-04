# Taylor Approximation

## Why It Matters for AI

Gradient descent takes a small step and trusts a simplified model of the loss. That simplified model is the **Taylor approximation**: replace a complicated function near the current point by a polynomial you can minimize exactly. This chapter supplies the approximation, and then shows the single most important derivation in training:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta)
$$

appearing as the exact solution of "minimize the local linear model."

## First Order (One Variable)

For differentiable $f$ and small $h$:

$$
\boxed{\;f(x + h) \approx f(x) + f'(x)h\;}
$$

The tangent line from chapter 02, used as a stand-in for the function. Validity: the error is $o(h)$ — it vanishes faster than $h$ itself, so halving $h$ more than halves the error.

**Worked example.** $f(x) = e^x$ at $x = 0$, $h = 0.1$:

$$
f(0) + f'(0)(0.1) = 1 + 0.1 = 1.1
\qquad\text{exact } e^{0.1} \approx 1.10517
$$

Error $\approx 0.005$ — about $5\%$ of $h^2$, consistent with the second-order error term below.

## Second Order

$$
f(x + h) \approx f(x) + f'(x)h + \frac{1}{2}f''(x)h^2
$$

The $\tfrac12$ comes from integrating the constant curvature twice (or expanding $e^h$, $\sin h$ etc. and matching). Error is now $o(h^2)$.

**Worked example — exact for quadratics.** $f(x) = x^2 - 4x + 5$ at $x = 1$, $h = 0.1$:

$$
f(1) = 2, \qquad f'(1) = 2(1) - 4 = -2, \qquad f''(x) = 2
$$

$$
f(1.1) \approx 2 + (-2)(0.1) + \tfrac12(2)(0.01) = 2 - 0.2 + 0.01 = 1.81
$$

Exact: $1.1^2 - 4(1.1) + 5 = 1.21 - 4.4 + 5 = 1.81$ ✓ — a quadratic matches its Taylor polynomial of degree 2 everywhere. (Also note: $f$ is minimized near $x = 2$; the negative slope at $x = 1$ says "go right" — direction of improvement, which is what descent follows.)

## Several Variables

For $J: \mathbb{R}^n \to \mathbb{R}$ and step $\Delta \in \mathbb{R}^n$:

**First order:**

$$
J(\theta + \Delta) \approx J(\theta) + \nabla J(\theta)^\top \Delta
$$

$\nabla J$ is the gradient (chapter 03); the product is a dot product — a number.

**Second order:**

$$
J(\theta + \Delta) \approx J(\theta) + \nabla J(\theta)^\top \Delta + \frac{1}{2}\Delta^\top H \Delta
$$

$H$ is the Hessian (chapter 04): $\Delta^\top H \Delta$ is the quadratic curvature term.

## Deriving Gradient Descent

**Problem.** Given $\theta$ and the first-order model, choose the step $\Delta$ that lowers the predicted loss most.

The model $g(\Delta) = J(\theta) + \nabla J^\top \Delta$ is *linear* in $\Delta$ — unconstrained, it decreases without bound (go far enough along $-\nabla J$ and $g \to -\infty$), which would mean nothing. Real steps must be small: restrict $\|\Delta\| \le \eta$ (the learning rate bounds the step).

**Minimize $g$ over the ball $\|\Delta\| \le \eta$.** Since

$$
\nabla J^\top \Delta = \|\nabla J\|\,\|\Delta\|\cos\theta
$$

is smallest when $\cos\theta = -1$ (opposite direction) and $\|\Delta\| = \eta$:

$$
\Delta^* = -\eta\,\frac{\nabla J}{\|\nabla J\|}
$$

The **direction** of steepest descent, full allowed length. Dropping the normalization (folding the scale into $\eta$, which also lets the step adapt to gradient magnitude) gives the standard update:

$$
\boxed{\;\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta_{\text{old}})\;}
$$

**Each symbol:**

| symbol | meaning |
|---|---|
| $\theta_{\text{old}}$ | current parameter vector (all weights at once) |
| $\nabla J(\theta_{\text{old}})$ | gradient of the loss at those parameters — direction of steepest *increase* |
| $\eta > 0$ | learning rate: how far to move per gradient (a modeling choice, fixed by the practitioner) |
| $-$ | descend, not ascend |
| $\theta_{\text{new}}$ | updated parameters |

**Why it decreases the loss** (the whole argument): substitute the update into the first-order model:

$$
J(\theta_{\text{new}}) \approx J(\theta) + \nabla J^\top(-\eta\nabla J)
= J(\theta) - \eta\|\nabla J\|^2
\;<\; J(\theta)
$$

for any $\eta > 0$ (when $\nabla J \ne \mathbf{0}$), **as long as the approximation holds** — i.e., $\eta$ small enough that second-order effects do not overturn the linear prediction. Too large an $\eta$: the linear model misses the curvature, the step overshoots, and the loss can rise. This is why learning rates must be chosen carefully — a fact derived, not asserted.

**One-dimensional sanity check.** $J(\theta) = \theta^2$, $\theta = 3$, $\eta = 0.1$:

$$
J'(3) = 6, \qquad \theta_{\text{new}} = 3 - 0.1(6) = 2.4
$$

$$
J: 9 \to 5.76 \quad\text{(decreased ✓)}
$$

Predicted by the model: $9 - 0.1 \cdot 36 = 5.4$ — close to the true $5.76$; the gap is exactly the discarded second-order term $\tfrac12 J'' \eta^2 \|...\|^2 = \tfrac12(2)(0.6)^2 = 0.36$, and $5.4 + 0.36 = 5.76$ ✓ (quadratic is exact to second order).

## What Second Order Buys

$\tfrac12 \Delta^\top H \Delta$ encodes curvature:

- $H \succ 0$ (positive definite): the model curves up — minimize it exactly and you get **Newton's method** $\Delta = -\eta H^{-1}\nabla J$, which rescales the gradient by inverse curvature (section 05 mentions it).
- Large eigenvalues of $H$ → sharp direction → smaller safe step along it; small eigenvalues → flat → larger step tolerable. Learning-rate methods (Adam etc.) adapt step sizes per-parameter using first-order statistics; full Newton curvature is usually too expensive to compute ($n \times n$ for $n$ = billions of parameters).

## Connection to AI

- **Gradient descent *is* the first-order Taylor model minimized.** The update rule is not folklore; it is the solution of a tiny optimization problem stated above.
- **Learning rate = trust region.** $\eta$ controls how far the linear model may be extrapolated — the same role as the step-size parameter in any Newton-style method.
- **First-order methods** (SGD, Adam) use $\nabla J$ only; **second-order** methods add $H$. The trade-off — cheap gradient versus expensive curvature — structures the whole optimizer landscape (section 05).
- **Directional derivatives** of chapter 03 are exactly $\nabla J^\top \Delta$ — the term this approximation is built from.

## Common Mistakes

- Applying the approximation with a large $\Delta$ and trusting the prediction — Taylor is *local*; the guarantee "loss decreases" fails outside the trust region.
- Forgetting $\tfrac12$ in the second-order term.
- Dropping the transpose: $\nabla J^\top \Delta$ is a scalar; $\Delta^\top H \Delta$ requires $H$ square.
- Believing $\theta_{\text{new}}$ always reduces the true $J$ — only the *approximation* is guaranteed to decrease; the actual loss needs $\eta$ small enough (or curvature controlled).
- Treating $-\eta\nabla J$ as the only descent direction — it is the steepest, but many directions descend; others may work better over multiple steps.

## Exercises

1. Approximate $f(x) = \sqrt{x}$ at $x = 4$ for $h = 0.1$ (first order). Exact value $\sqrt{4.1} \approx 2.02485$ — compute the error.
2. Use the second-order expansion for $f(x) = x^2 - 4x + 5$ at $x = 1$ to predict $f(1.2)$; verify exactly.
3. Write the first-order Taylor expansion of $J(\theta) = \theta_1^2 + 2\theta_2^2$ at $\theta = (1, 1)$ in the direction $\Delta = (0.1, -0.1)$; evaluate the approximation and the true $J(1.1, 0.9)$.
4. Take one gradient-descent step on $J(\theta) = \theta^2$ from $\theta = 3$ with $\eta = 0.05$; compare the resulting loss with the prediction from the linear model.
5. Show algebraically that $J(\theta - \eta\nabla J) \approx J(\theta) - \eta\|\nabla J\|^2$ using the first-order expansion.
6. In the derivation, why must the step be constrained ($\|\Delta\| \le \eta$) before minimizing the linear model?
