# Stochastic Gradient Descent

## Why It Matters for AI

A full gradient costs one pass over the entire dataset — for a billion-example corpus and a model with billions of parameters, computing $\nabla J$ exactly is impossible. **Stochastic gradient descent (SGD)** replaces the exact average with a random sample of examples, turning each step's cost from "all the data" into "one (or a few) examples". The price is noise; the payoff is tractability. This chapter shows why the noisy estimate is *unbiased*, what the noise does to training, and how minibatching splits the difference.

## From Full Batch to Stochastic

The objective is an average (chapter 01):

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^{n} L(\theta; x_i, y_i), \qquad
\nabla J = \frac{1}{n}\sum_{i=1}^{n} \nabla L_i
$$

- **Full-batch gradient descent:** use $\nabla J$ exactly — cost $O(n)$ per step, deterministic path.
- **SGD:** pick an index $I$ uniformly at random from $\{1,\dots,n\}$ and use $\nabla L_I(\theta)$ alone — cost $O(1)$ per step, random path.
- **Minibatch:** use a random batch $B$ and average $|B|$ gradients — cost $O(|B|)$, the practical compromise (GPUs process batches in parallel, so throughput improves with batch size up to a point).

## Why the Single-Example Gradient Works: Unbiasedness

The key computation (one line, by linearity of expectation):

$$
\mathbb{E}_I\big[\nabla L_I(\theta)\big] = \frac{1}{n}\sum_{i=1}^{n} \nabla L_i(\theta) = \nabla J(\theta)
$$

The random gradient is an **unbiased estimate** of the true gradient: on average it points the right way, even though each individual step does not. Same principle as sampling a mean in section 04 — statistics doing algorithmic work.

**Worked example.** $n = 3$: $L_i(w) = \tfrac12 (w - y_i)^2$ with $y = (1, 3, 5)$.

$$
\nabla J(w) = \tfrac13\big[(w-1) + (w-3) + (w-5)\big] = w - 3
$$

(check: the full-batch optimum is $w^* = 3 = \bar y$ — the sample mean, exactly as MLE in section 04.)

Single-example gradients at $w = 0$: $-1,\ -3,\ -5$; their average $= -3 = \nabla J(0)$ ✓ — unbiased.

## The Noise, and What It Does

Each step $w \leftarrow w - \eta\,g_I$ has $g_I = \nabla J + \xi$ with a zero-mean random "noise" term $\xi$ of variance that does not vanish (it shrinks only as the examples agree near the optimum).

Consequences:

1. **Cost per step:** $O(1)$ instead of $O(n)$ — orders of magnitude cheaper; progress per unit compute is usually far better early in training.
2. **Path is not monotone:** individual steps can *increase* $J$ — the descent guarantee of chapter 02 applies only to the expectation $\mathbb{E}[g] = \nabla J$, not each realization.
3. **Noise floors the convergence:** near the optimum the gradient is tiny but the noise $\eta\xi$ is not — iterates bounce around $w^*$ in a neighborhood of radius $O(\eta)$. Converging *exactly* requires decaying $\eta \to 0$ over training.
4. **Escaping bad points:** the same jitter pushes training out of saddle points and shallow local minima that trap exact gradient descent (section 02's critical points) — a practical blessing of the noise.

**Convergence in expectation** holds under standard conditions on the schedule (roughly: $\sum_k \eta_k = \infty$ so you can still travel arbitrarily far, and $\sum_k \eta_k^2 < \infty$ so accumulated noise stays bounded). We state this as a fact about SGD theory without proving it; the intuition is: infinite total step size, vanishing total noise.

**Worked loop — SGD on the toy objective.**

```python
import random
random.seed(0)
ys = [1.0, 3.0, 5.0]
w = 0.0
eta = 0.1
for epoch in range(6):
    for i in random.sample(range(3), 3):      # one pass = one epoch, shuffled
        w -= eta * (w - ys[i])                # gradient of L_i = 0.5(w - y_i)^2
    print(f"epoch {epoch}: w = {w:.4f}")
# epoch 0: w = 0.7930, epoch 1: w = 1.3711, ... epoch 5: w = 2.4992
```

One **epoch** = one full pass over the data ($n$ stochastic steps). The path zig-zags (each single-example gradient points at a different $y_i$) but drifts toward the optimum $3$: after 30 epochs the same loop reaches $w \approx 2.99$, bouncing in a small neighborhood of it — behavior predicted by points 2–3 (the residual gap and jitter are the $\eta$-sized noise floor).

## Choosing the Batch

| batch | gradient cost | noise | hardware fit |
|---|---|---|---|
| 1 | cheapest | max | poor (no parallelism) |
| 64–4096 | moderate | moderate | good (vectorized) |
| all $n$ | per-step worst | none | needs huge steps or infrequent updates |

- **Throughput:** GPUs amortize arithmetic over the batch; too-small batches waste parallelism, too-large ones buy diminishing noise reduction for linearly growing cost.
- **Larger batch ⇒ smaller noise ⇒ often larger learning rate** (a common scaling heuristic: keep the *total* gradient magnitude per update roughly stable).
- **Shuffling** matters: feeding data in fixed order lets the biased-looking sequence of *particular* indices (not their expectation) dominate — shuffle each epoch.
- **Learning-rate schedule** (decay/warmup, chapter 02) handles the early "large steps over noisy terrain" and later "small steps to settle" phases.

## Connection to AI

- Every deep-learning trainer loops: sample batch → forward pass → backward pass (compute $\nabla L$ for the batch) → optimizer step. SGD *is* the loop.
- **Mini-batch gradients are the reason epochs exist** as a unit of budget: "10 epochs on ImageNet" means $10 \times n/|B|$ stochastic steps.
- Adaptive methods (Adam etc.) keep the stochastic estimate and precondition it per-parameter — beyond this section, but they inherit the unbiasedness analysis.
- **Noise as regularization:** the stochasticity itself mildly regularizes (large-batch models often fit sharper minima) — connects to chapter 05.
- **Large-batch research** (linear scaling rules, warmup) is exactly this chapter's trade-off quantified at scale.

## Common Mistakes

- Averaging a minibatch loss *and* dividing by dataset size again (or vice versa) — the gradient is scaled by $1/|B|$ per batch, consistently; a wrong factor rescales your effective learning rate.
- No shuffling: systematic order (sorted labels, grouped classes) turns random indices into biased ones.
- Assuming each step decreases the training loss — SGD steps regularly increase it; judge by epoch-level trends.
- Keeping $\eta$ fixed forever: iterates never settle below the $O(\eta)$ noise floor.
- Using the whole dataset every step and wondering why training crawls — the definition of needing SGD.
- Comparing runs with different batch sizes at fixed $\eta$ without rescaling — effective step size per example changed.

## Exercises

1. Compute $\nabla J$ exactly and each $\nabla L_i$ for $n=2$: $L_i(w) = (w - y_i)^2$, $y = (0, 4)$; then verify unbiasedness numerically at $w = 1$ (average the two single-example gradients).
2. What is the full-batch optimum $w^*$ for that objective, and which statistic of $\{0, 4\}$ is it?
3. State in one sentence why $\mathbb{E}[\nabla L_I] = \nabla J$ (the two quantities you are equating).
4. In the worked Python loop, what would you expect if `eta` were $0.01$ instead of $0.1$? (Two symptoms.)
5. An epoch is 64,000 steps. What dataset size and batch size is that? (One sensible pair.)
6. Why can SGD escape a saddle point when full-batch descent cannot? (One sentence, key word: noise.)
