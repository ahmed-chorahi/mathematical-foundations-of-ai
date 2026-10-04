# MAP Estimation

## Why It Matters for AI

Maximum likelihood asks only "what fits the data best?". With little data or many parameters, that answer overfits. **Maximum a posteriori** estimation adds a **prior** — a belief about plausible parameters before seeing data — turning estimation into Bayes' theorem (chapter 02 of section 03) applied to parameters. The Gaussian prior produces exactly the $L_2$ penalty you know as weight decay; the Laplace prior produces $L_1$. Regularization, viewed from this angle, is not a trick — it is stated prior knowledge.

## Bayes for Parameters

Section 03: $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$. Substitute $A = \theta$ (parameters), $B = D$ (data):

$$
\boxed{\;P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)}\;}
$$

| term | name | role |
|---|---|---|
| $P(\theta \mid D)$ | **posterior** | belief about parameters *after* data |
| $P(D \mid \theta)$ | **likelihood** | how well parameters fit (chapter 03) |
| $P(\theta)$ | **prior** | belief *before* data |
| $P(D)$ | **evidence** | normalizing constant: $\int P(D\mid\theta)P(\theta)\,d\theta$ |

**MAP estimate** = the posterior's mode (most probable parameter value):

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta \; P(\theta \mid D)
$$

Since $P(D)$ does not depend on $\theta$, it drops out of the argmax:

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta \Big[\log P(D \mid \theta) + \log P(\theta)\Big]
$$

$$
\text{MAP} = \text{MLE} + \text{prior term}
$$

The likelihood term is the log-likelihood from chapter 03; $\log P(\theta)$ is the new ingredient.

## Worked Example: Gaussian Prior → L2 Penalty

**Setup.** Each parameter component has prior $\theta_j \sim N(0, \tau^2)$ independently. The log-prior:

$$
\log P(\theta) = \sum_j \left[-\frac{1}{2}\ln(2\pi\tau^2) - \frac{\theta_j^2}{2\tau^2}\right]
= \text{const} - \frac{1}{2\tau^2}\|\theta\|_2^2
$$

Maximizing posterior = maximizing log-likelihood $- \frac{1}{2\tau^2}\|\theta\|^2$, equivalently **minimizing**:

$$
J(\theta) = \underbrace{-\log P(D \mid \theta)}_{\text{loss}} + \underbrace{\frac{1}{2\tau^2}\|\theta\|_2^2}_{\text{penalty}}
$$

This is exactly **ridge regression / weight decay** — the $L_2$ regularized objective of section 05. Two knobs, one interpretation:

| frequentist name | Bayesian name |
|---|---|
| regularization strength $\lambda = \frac{1}{2\tau^2}$ | inverse prior variance $1/\tau^2$ |
| small $\tau$ (tight prior) | strong penalty, weights pushed hard toward 0 |
| large $\tau$ (vague prior) | weak penalty → plain MLE |
| $\tau \to \infty$ | flat prior, penalty vanishes, MAP = MLE |

**Strength of the prior is its variance** — a clean quantitative statement of "how much do you believe your prior".

**Numerical example (scalar).** Suppose the likelihood term is $(\theta - 10)^2$ (the data favors $\theta \approx 10$) and the Gaussian prior contributes $\lambda\theta^2$ (where $\lambda = \sigma^2/\tau^2$ absorbs both scales, as in the boxed objective above). With $\lambda = 1$:

$$
\min_\theta\; (\theta - 10)^2 + \theta^2
$$

Derivative: $2(\theta - 10) + 2\theta = 0 \Rightarrow \theta = 5$ — shrunk halfway from the data's suggestion toward the prior mean $0$. Larger $\lambda$ shrinks further; $\lambda \to 0$ returns $10$.

## Laplace Prior → L1 Penalty

Prior $\theta_j \sim \text{Laplace}(0, b)$: density $\propto \exp(-|\theta_j|/b)$, so $\log P(\theta) \propto -\frac{1}{b}\|\theta\|_1$. MAP objective:

$$
J(\theta) = -\log P(D \mid \theta) + \frac{1}{b}\|\theta\|_1
$$

the **lasso** penalty. Difference from $L_2$: the $L_1$ ball's corners lie on axes, so its optimum tends to set some parameters *exactly* to zero — **sparsity**, interpreted as automatic feature selection (section 05 develops the geometry; here the Bayesian origin is the point).

## MAP vs MLE vs Full Bayes

| approach | what it returns | uses prior how |
|---|---|---|
| **MLE** | a point $\hat\theta$ maximizing likelihood | ignores it |
| **MAP** | a point maximizing posterior | as a term in an optimization |
| **Full Bayes** | a distribution $P(\theta \mid D)$ | propagates uncertainty everywhere |

MAP is a point estimate of a posterior — often enough for practice, but it hides uncertainty (it reports one answer with no error bars). Full Bayesian inference integrates over $\theta$ ($\mathbb{E}_{\theta \sim P(\theta|D)}[\cdot]$) — more honest, more expensive.

## Connection to AI

- **Weight decay = Gaussian prior.** Saying "weights should be near zero, with spread $\tau$" is a modeling statement; training with $L_2$ penalty implements it. The two descriptions are the same mathematics.
- **Lasso = Laplace prior** → sparse models, feature selection.
- **Why regularization helps:** chapter 02's variance reduction — the prior drags estimates toward plausible values, increasing bias slightly and cutting variance more (section 05 quantifies when this wins).
- **Early stopping** has a similar effect (stops before the fit chases noise) and can be shown to correspond to an implicit prior — a research-level statement, but the direction is the same: extra assumptions reduce variance.
- **Hyperparameter $\lambda$ / $\tau$:** chosen by validation (empirical Bayes) or set by domain knowledge — a choice of prior strength, for better or worse.
- **Bayesian neural networks** push this to the limit: put priors on all weights and approximate the posterior — at the cost of significant computation (out of scope here beyond the concept).

## Common Mistakes

- Maximizing $P(D \mid \theta)P(\theta)$ *with* the evidence term and calling it MAP — $P(D)$ is constant in $\theta$; include or drop it, the argmax is identical (but do not drop it if you need actual posterior *values*).
- Forgetting to convert "regularization strength" into "prior variance" and back — $\lambda = 1/(2\tau^2)$, not $\tau$.
- Treating the prior as decoration — a strong wrong prior biases the fit badly; MAP inherits both the data's noise and the prior's error.
- Confusing MAP (a point) with the posterior (a distribution).
- Assuming MAP always beats MLE — with abundant clean data the prior's contribution fades ($P(D)$ dominates); with little data it dominates.

## Exercises

1. Write the MAP objective (up to constants in $\theta$) for likelihood $P(D \mid \theta)$ and prior $P(\theta)$ — i.e., what is maximized.
2. Data: 3 heads in 5 Bernoulli flips; prior $p \sim \text{Uniform}(0,1)$. What is the MAP estimate, and why does the uniform prior change nothing?
3. Same data; prior concentrated near $p = 0.5$ (say, proportional to $p(1-p)$ — a Beta(2,2)-shaped belief). Write the log-posterior and find $\hat p_{\text{MAP}}$ by differentiating. (Expect shrinkage toward 0.5: confirm numerically that the answer lies between 0.5 and 0.6.)
4. For the scalar example in the chapter, solve $\min_\theta (\theta - 10)^2 + \lambda\theta^2$ in general (answer in terms of $\lambda$), and evaluate at $\lambda = 0$, $\lambda = 1$, $\lambda = 9$.
5. State the MAP objective for an $L_1$ penalty and name the corresponding prior family.
6. Explain in one sentence why $\tau \to \infty$ (infinitely vague Gaussian prior) recovers MLE.
