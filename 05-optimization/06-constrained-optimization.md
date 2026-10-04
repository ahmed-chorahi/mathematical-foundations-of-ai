# Constrained Optimization and Lagrange Multipliers

## Why It Matters for AI

Sometimes the best solution must satisfy rules: a budget, a margin requirement, a normalization, a feasibility condition. **Constrained optimization** answers: how do you minimize an objective *subject to* constraints? The method — build a **Lagrangian**, solve stationarity — is short, and it is the doorway to the support vector machine, to duality, and to how modern ML handles constraints (projection steps, penalty methods).

## The Problem Forms

$$
\min_{\theta \in \mathbb{R}^d} f(\theta)
\quad\text{subject to}\quad
\begin{cases}
g_j(\theta) = 0, & j = 1,\dots,m \quad\text{(equality)}\\[2pt]
h_k(\theta) \le 0, & k = 1,\dots,p \quad\text{(inequality)}
\end{cases}
$$

- The **feasible set** $\{\theta : g_j = 0,\ h_k \le 0\}$ is where solutions may live; unconstrained critical points outside it are irrelevant.
- If $\theta^*$ is unconstrained-optimal and feasible, constraints do nothing; otherwise the constraints *bind* — they push the answer to the set's boundary.

## Equality Constraints: The Lagrangian

Define the **Lagrangian** (calligraphic $\mathcal{L}$ to distinguish from the loss $L$):

$$
\mathcal{L}(\theta, \lambda) = f(\theta) + \sum_{j=1}^{m} \lambda_j\, g_j(\theta)
$$

$\lambda_j$ are **Lagrange multipliers** — one per equality constraint; they are variables, not fixed numbers.

**Method of Lagrange multipliers** — solve the $d + m$ stationarity equations:

$$
\boxed{\;\nabla_\theta \mathcal{L} = \mathbf{0}
\qquad\Longleftrightarrow\qquad
\nabla f(\theta) + \sum_j \lambda_j \nabla g_j(\theta) = \mathbf{0},
\qquad g_j(\theta) = 0\;}
$$

**Why it works (geometric reading).** Suppose $\theta^*$ minimizes $f$ subject to $g = 0$, and $\nabla g(\theta^*) \ne \mathbf{0}$. If $\nabla f$ were *not* parallel to $\nabla g$, you could step along the component of $-\nabla f$ tangent to the constraint surface ($\nabla g^\top v = 0$, so the constraint stays satisfied to first order) and strictly decrease $f$ — contradicting optimality. Hence $\nabla f = -\lambda \nabla g$: the constraint's normal direction "absorbs" part of the objective's gradient, and $\lambda$ measures how strongly the constraint resists.

Second-order detail: optimality also requires the Hessian of $\mathcal{L}$ restricted to directions tangent to the constraints to be positive semi-definite (the "Lagrangian is convex along the feasible directions" condition). We check this only informally below.

**Worked example — nearest point to the origin on a line.**

$$
\min\; f(x,y) = x^2 + y^2
\quad\text{s.t.}\quad
g(x,y) = x + y - 2 = 0
$$

$$
\mathcal{L} = x^2 + y^2 + \lambda(x + y - 2)
$$

$$
\frac{\partial \mathcal{L}}{\partial x} = 2x + \lambda = 0,\quad
\frac{\partial \mathcal{L}}{\partial y} = 2y + \lambda = 0,\quad
\frac{\partial \mathcal{L}}{\partial \lambda} = x + y - 2 = 0
$$

From the first two: $x = y = -\lambda/2$ (gradients parallel ✓). Constraint: $2x = 2 \Rightarrow x = y = 1$, hence $\lambda = -2$.

**Solution:** $\theta^* = (1,1)$, minimum value $f = 2$, multiplier $|\lambda| = 2$.

*Sanity checks.* Unconstrained minimum is $(0,0)$ — infeasible ($0 + 0 \ne 2$) ✓. The constrained answer lies on the line, and $\nabla f = (2,2) = -2\,(1,1) = -\lambda\nabla g$ ✓.

**Sensitivity reading of $\lambda$:** the multiplier's magnitude is the rate at which the optimum value changes if you relax the constraint: on $x + y = 2 + \varepsilon$ the optimum moves to $(1+\varepsilon/2,\,1+\varepsilon/2)$, value $2(1+\varepsilon/2)^2 = 2 + 2\varepsilon + O(\varepsilon^2)$, so $\frac{d}{d\varepsilon}\big|_0 = 2 = -\lambda$ — in the $\mathcal{L} = f + \lambda g$ convention the value's sensitivity is $-\lambda$. Multipliers are shadow prices.

## Inequality Constraints: KKT (Sketch)

For $h_k(\theta) \le 0$, multipliers $\mu_k \ge 0$ satisfy:

$$
\nabla f + \sum_j \lambda_j \nabla g_j + \sum_k \mu_k \nabla h_k = \mathbf{0},
\qquad
\mu_k h_k(\theta) = 0 \ \ \text{(complementary slackness)},
\qquad
h_k(\theta) \le 0
$$

plus feasibility and the tangent-space second-order condition. The new ingredient is **complementary slackness**: either the constraint is *active* ($h_k = 0$, $\mu_k$ may be positive — it pushes) or *inactive* ($h_k < 0$, then $\mu_k = 0$ — it doesn't). A $\mu_k > 0$ for an inactive constraint, or a negative $\mu_k$, signals a wrong solution.

*Worked activation check.* $\min_x \tfrac12 x^2$ s.t. $x \ge 1$ (i.e., $h = 1 - x \le 0$). Unconstrained $x = 0$ violates; so the constraint must be active: $x^* = 1$, and stationarity $x + \mu(-1) = 0 \Rightarrow \mu = 1 > 0$ ✓ consistent (with $\mathcal{L} = f + \mu h$). The inequality genuinely binds.

The full set of conditions — feasibility, stationarity, primal/dual feasibility, complementary slackness — is the **Karush–Kuhn–Tucker (KKT)** system; for convex $f, h_k$ and affine $g_j$, KKT is *sufficient* for global optimality (the convex-constrained analogue of chapter 03's guarantee).

## Connection to AI: The Support Vector Machine

The canonical ML instance. Binary classification with hard margin: find $w$ with as large a margin as possible subject to every point being on the correct side of a fixed distance:

$$
\min_w \ \tfrac12\|w\|^2
\quad\text{s.t.}\quad
y_i\,(w^\top x_i + b) \ge 1, \quad i = 1,\dots,n
$$

(the margin between the two class-boundary hyperplanes equals $2/\|w\|$ — shrinking $\|w\|$ widens it; the constant 1 fixes the scale).

Lagrangian with multipliers $\alpha_i \ge 0$:

$$
\mathcal{L} = \tfrac12\|w\|^2 - \sum_i \alpha_i\big(y_i(w^\top x_i + b) - 1\big)
$$

Stationarity in $w, b$ gives $w = \sum_i \alpha_i y_i x_i$ and $\sum_i \alpha_i y_i = 0$; complementary slackness says points with $\alpha_i > 0$ sit exactly on the margin — the **support vectors**. Everything about the SVM (its solution expressed through a few points, its dual form, kernelization) follows from this Lagrangian — the pattern of this chapter, applied at scale. Soft-margin SVM ($\xi_i \ge 0$ slack variables, penalty $C$) adds inequality constraints of exactly the KKT kind.

Lagrange machinery also appears in: **dual descent methods**, **normalizing outputs** under constraints (projection), **RLHF-style optimization with KL constraints** (in section 10's orbit), and anytime a quantity is optimized "under a budget".

## Common Mistakes

- Forgetting to include the constraint equations among the unknowns — solving only $\nabla_\theta \mathcal{L} = 0$ for $\theta, \lambda$ without $g_j(\theta) = 0$ leaves a family of candidates.
- Sign conventions: $\mathcal{L} = f + \lambda g$ with $\nabla f + \lambda\nabla g = 0$, or $f - \lambda g$ — both fine, but mixing them flips signs; KKT demands $\mu \ge 0$ only in the stated convention.
- Using a multiplier for an inactive inequality (violates complementary slackness).
- Missing multiple candidates: stationarity can yield several $\theta$'s (e.g., intersecting constraints) — compare $f$ values, and check the second-order condition.
- Treating constrained and unconstrained answers alike when the constraint is inactive — then $\lambda = 0$ and the methods coincide (a useful check: if your $\lambda \ne 0$, the constraint *did* matter).
- Assuming KKT sufficiency without convexity — for non-convex $f$, KKT is necessary (under regularity) but only a local statement.

## Exercises

1. Set up the Lagrangian and stationarity system for: $\min\ x^2 + 4y^2$ s.t. $x + y = 1$.
2. Solve it: find $(x, y, \lambda)$.
3. For the chapter's worked example, find the solution on $x + y = 2 + \varepsilon$. Differentiate the optimum value at $\varepsilon = 0$ and check it equals $-\lambda$ in the chapter's $\mathcal{L} = f + \lambda g$ convention — the shadow-price claim.
4. $\min \tfrac12\|w\|^2$ s.t. $w \ge 3$ (one-dimensional, $h(w) = 3 - w \le 0$). Find $w^*$ and check complementary slackness and $\mu \ge 0$.
5. In the SVM Lagrangian, from stationarity in $b$: show $\sum_i \alpha_i y_i = 0$.
6. Why is the unconstrained minimum of $f$ the answer when it happens to be feasible? (One sentence with the word $\lambda$.)
