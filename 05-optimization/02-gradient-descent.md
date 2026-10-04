# Gradient Descent

## Why It Matters for AI

Gradient descent is the algorithm that trains nearly every model you use. One update — subtract a scaled gradient — repeated until the loss stops falling. This chapter derives the rule carefully (it was previewed in section 02, chapter 06), explains every symbol, works the iterations by hand, runs the loop in pure Python, and shows what goes wrong when the learning rate is chosen badly.

## The Update Rule

$$
\boxed{\;\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta_{\text{old}})\;}
$$

| symbol | meaning |
|---|---|
| $\theta$ | parameter vector — every weight and bias of the model in one symbol ($\theta \in \mathbb{R}^d$) |
| $J(\theta)$ | objective (average loss over data — chapter 01) |
| $\nabla J(\theta)$ | gradient: vector of all partial derivatives, pointing in the direction of steepest *increase* |
| $\eta > 0$ | **learning rate** — step size; a hyperparameter, fixed by you |
| $-$ | descend (against the gradient) |

Component-wise, the update is $d$ independent-looking scalar updates executed in parallel:

$$
\theta_j \leftarrow \theta_j - \eta\,\frac{\partial J}{\partial \theta_j}, \qquad j = 1, \ldots, d
$$

## Why This Rule Works (Derivation)

From section 02's Taylor chapter — repeated here in full because it is the foundation.

**Step 1 — linearize.** For a small step $\Delta$:

$$
J(\theta + \Delta) \approx J(\theta) + \nabla J(\theta)^\top \Delta
$$

**Step 2 — choose $\Delta$ to minimize the linear model** over $\|\Delta\| \le \eta$. Since $\nabla J^\top \Delta = \|\nabla J\|\|\Delta\|\cos\varphi$, the minimum over the allowed ball is at $\varphi = \pi$ (opposite direction) with full length:

$$
\Delta = -\eta\,\frac{\nabla J}{\|\nabla J\|}
$$

The steepest descent direction. Dropping the normalization (folding magnitude into $\eta$) gives the standard form $\Delta = -\eta\nabla J$.

**Step 3 — verify descent.** Substitute:

$$
J(\theta_{\text{new}}) \approx J(\theta) - \eta\|\nabla J\|^2 < J(\theta)
$$

whenever $\nabla J \ne \mathbf{0}$ and $\eta$ is small enough that the linear model is accurate over the step. The inequality *fails* if $\eta$ is too large — the entire practical art of learning-rate selection.

**Stopping:** $\|\nabla J\|$ near zero (a critical point — chapter 07 of section 02), a fixed iteration budget, or no progress on validation data.

## Worked Example: Iterations by Hand

$$
J(\theta) = (\theta - 4)^2, \qquad \theta_0 = 0, \qquad \eta = 0.1
$$

Gradient: $J'(\theta) = 2(\theta - 4)$.

| iteration | $\theta$ | $J'(\theta) = 2(\theta-4)$ | update: $\theta - 0.1 \cdot J'$ | new $\theta$ |
|---|---|---|---|---|
| 0 | 0 | −8 | $0 - 0.1(-8)$ | 0.8 |
| 1 | 0.8 | −6.4 | $0.8 + 0.64$ | 1.44 |
| 2 | 1.44 | −5.12 | $1.44 + 0.512$ | 1.952 |
| 3 | 1.952 | −4.096 | … | 2.3616 |

Loss: $16 \to 10.24 \to 6.55 \to \ldots \to 0$ — monotone decrease toward $\theta^* = 4$.

The exact sequence: $\theta_{k+1} = \theta_k - 0.2(\theta_k - 4) = 0.8\theta_k + 0.8$, so $\theta_k = 4(1 - 0.8^k)$ — geometric convergence with ratio $0.8$: each step closes $20\%$ of the remaining gap.

```python
def gradient_descent(grad, theta0, eta, steps):
    theta = theta0
    for _ in range(steps):
        theta = theta - eta * grad(theta)
    return theta

grad = lambda t: 2 * (t - 4)
print(gradient_descent(grad, theta0=0.0, eta=0.1, steps=10))
# 2.74999366 ...  approaching 4
```

After 10 steps the exact formula gives $\theta_{10} = 4(1 - 0.8^{10}) \approx 4(1 - 0.1074) \approx 3.57$ — the loop above reproduces this value.

## The Learning Rate

$\eta$ is the only knob in the rule — and decisive.

**Too small.** Progress is slow: closing $20\%$ of the gap took $\eta = 0.1$'s ratio; with $\eta = 0.001$ the ratio becomes $0.998$ — hundreds of iterations per percent of progress. Training time grows linearly in $1/\eta$ (roughly).

**Too large — divergence.** For $J(\theta) = \frac{\lambda}{2}\theta^2$ (curvature $\lambda$), the update gives

$$
\theta_{\text{new}} = \theta - \eta\lambda\theta = (1 - \eta\lambda)\theta
$$

| $1 - \eta\lambda$ | behavior |
|---|---|
| $0 < \cdot < 1$ | monotone convergence |
| $-1 < \cdot < 0$ | oscillating convergence (overshoot each side, shrinking) |
| $|1 - \eta\lambda| \ge 1$ | divergence — loss grows without bound |

Convergence for this quadratic requires $0 < \eta < 2/\lambda$. **The safe step size depends on curvature** — sharp directions ($\lambda$ large) tolerate less. Standardizing features (section 04) evens out curvature; adaptive optimizers (beyond this section) adjust per-direction.

**Rule of thumb in practice:** start around $10^{-3}$–$10^{-1}$ depending on method, monitor the loss curve, reduce when oscillating or plateauing, warm up when unstable. Empirical, but the stability condition above is the theory behind the symptoms.

## Gradient of the Average Loss

Since $J = \frac1n\sum_i L_i$, linearity gives

$$
\nabla J = \frac{1}{n}\sum_{i=1}^n \nabla L_i
$$

Full-batch gradient descent uses the exact average each step. Chapter 04 shows how estimating this sum with a minibatch changes the algorithm.

## A Two-Parameter Sketch

$J(w, b) = (wx + b - y)^2$ at $x = 1$, $y = 3$, $(w, b) = (1, 1)$:

$$
\frac{\partial J}{\partial w} = 2(\hat y - y)x = 2(2-3)(1) = -2, \qquad
\frac{\partial J}{\partial b} = 2(\hat y - y) = -1
$$

With $\eta = 0.1$: $w \leftarrow 1 + 0.2 = 1.2$, $b \leftarrow 1 + 0.1 = 1.1$ — both move toward the perfect-fit line $wx + b = 3$ (at $w = 1.2, b = 1.1$: prediction $2.3$, error $-0.7$ — improved from $-1$).

## Connection to AI

- **Training loops are this function.** Every framework's `optimizer.step()` applies a variant of the line above; "epochs" and "iterations" count its executions.
- **Learning-rate schedules** (decay, warmup) modulate $\eta$ over time — responses to the stability analysis: large early steps for progress, smaller later for settling.
- **The derivation constrains what optimizers can do:** any first-order method is a (possibly preconditioned, possibly noisy) version of $-\eta\nabla J$; momentum and Adam change the *effective* direction/step (chapter 04's SGD family).
- **Why gradients are precious:** one backward pass produces $\nabla J$ for millions of parameters simultaneously — the information efficiency that makes training feasible.
- **Critical points** reached when gradients vanish — saddles included (section 02, chapter 07); noise (chapter 04) is what escapes them.

## Common Mistakes

- Using $\eta$ too large after changing model/data scale — gradients scale with loss magnitudes; a previously stable $\eta$ may now diverge.
- Forgetting that $\nabla J$ must be evaluated **at the current $\theta$** before updating (reusing stale gradients is a different method).
- Mixing gradient *ascent* signs for maximization problems — maximize $J$ by $+\eta\nabla J$, or minimize $-J$.
- Assuming monotone decrease for every $\eta > 0$ — only within the trust region; the quadratic analysis shows failure explicitly.
- Concluding convergence = global optimum — without convexity (chapter 03), you reach *a* critical point.

## Exercises

1. Run two gradient-descent steps on $J(\theta) = \theta^2$ from $\theta_0 = 1$ with $\eta = 0.5$; compute both new $\theta$ values and the loss after each.
2. For $J(\theta) = \frac{1}{2}\lambda\theta^2$ with $\lambda = 4$, find the range of $\eta$ for convergence using the factor $1 - \eta\lambda$; then classify the behavior at $\eta = 0.5$.
3. Take one step for $J(w) = (2w - 6)^2$ from $w = 0$, $\eta = 0.05$. ($J'(w) = 2(2w-6)\cdot 2$.)
4. Compute the full gradient of $J(w,b) = (wx+b-y)^2$ symbolically, then evaluate at $x = 2$, $y = 10$, $w = 1$, $b = 1$, $\eta = 0.01$; give the updated $(w, b)$.
5. Explain in two sentences why the safe learning rate depends on curvature, referencing $1 - \eta\lambda$.
6. Modify the chapter's Python function to stop when $|grad(theta)| < 10^{-6}$; how many iterations does it need from $\theta_0 = 0$, $\eta = 0.1$? (Run it, or estimate from the geometric rate.)
