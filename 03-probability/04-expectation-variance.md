# Expectation, Variance, and Covariance

## Why It Matters for AI

Training minimizes an *expected* loss; uncertainty is measured by *variance*; feature relationships and embedding geometry are described by *covariance*. These three summaries turn a full distribution into actionable numbers — and their linearity properties are what make training formulas tractable.

## Expectation (Expected Value)

For a discrete random variable with PMF $p$:

$$
\mathbb{E}[X] = \sum_x x\,p(x)
$$

The probability-weighted average of the possible values — weight each outcome by how likely it is, add up.

For continuous $X$ with density $f$: $\mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x)\,dx$.

**Worked example — fair die:**

$$
\mathbb{E}[X] = 1\cdot\tfrac16 + 2\cdot\tfrac16 + 3\cdot\tfrac16 + 4\cdot\tfrac16 + 5\cdot\tfrac16 + 6\cdot\tfrac16
= \frac{21}{6} = 3.5
$$

Not an outcome you can roll — the long-run average of many rolls. That is what "expected" means: balance point of the distribution.

**Worked example — Bernoulli($p$):**

$$
\mathbb{E}[X] = 1\cdot p + 0\cdot(1-p) = p
$$

The success probability *is* the expected value of the indicator.

### Properties (linearity)

For constants $a, b$ and random variables $X, Y$ (any dependence structure):

$$
\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b
$$

$$
\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]
$$

**No independence required** — linearity of the sum $\sum_x (x+y)p(x) = \sum x p(x) + \sum y p(y)$ is just algebra. This is why the expected loss of a dataset decomposes:

$$
\mathbb{E}_{\text{examples}}[L] = \frac{1}{n}\sum_{i=1}^{n} L_i
\quad\Rightarrow\quad
\nabla\,\mathbb{E}[L] = \frac{1}{n}\sum_i \nabla L_i
$$

(differentiating a sum term by term) — the mathematical license for minibatch training (section 05).

**Non-linear functions:** $\mathbb{E}[g(X)] \ne g(\mathbb{E}[X])$ in general. (Roll a die: $\mathbb{E}[X^2] = 91/6 \approx 15.17 \ne (3.5)^2 = 12.25$.) Expectation passes through linear operations only.

## Variance and Standard Deviation

**Variance** measures spread around the mean:

$$
\operatorname{Var}(X) = \mathbb{E}\left[(X - \mu)^2\right], \qquad \mu = \mathbb{E}[X]
$$

Equivalent (and easier) computational form:

$$
\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

*Derivation:* expand and use linearity:
$$
\mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2 - 2\mu X + \mu^2] = \mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - \mu^2
$$

**Standard deviation** $\sigma = \sqrt{\operatorname{Var}(X)}$ — same units as $X$ (variance is squared units, hard to read directly).

**Worked example — fair die:** $\mu = 3.5$.

$$
\mathbb{E}[X^2] = \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.167
$$

$$
\operatorname{Var}(X) = \frac{91}{6} - \left(\frac{7}{2}\right)^2 = \frac{182 - 147}{12} = \frac{35}{12} \approx 2.917
\qquad \sigma = \sqrt{\tfrac{35}{12}} \approx 1.708
$$

**Bernoulli($p$):** $\mathbb{E}[X^2] = p$ (since $1^2 = 1$), so

$$
\operatorname{Var}(X) = p - p^2 = p(1-p)
$$

Maximum at $p = 0.5$ ($0.25$), zero at the extremes — a fair coin is the most uncertain; a certain outcome has no variance.

**Scaling:** $\operatorname{Var}(aX + b) = a^2\operatorname{Var}(X)$ (the shift $b$ does not move the spread; scaling $a$ enters squared).

## Covariance

Two random variables: do they move together?

$$
\operatorname{Cov}(X, Y) = \mathbb{E}\left[(X - \mu_X)(Y - \mu_Y)\right] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

*The second form's derivation* mirrors the variance case: expand $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)] = \mathbb{E}[XY] - \mu_Y\mathbb{E}[X] - \mu_X\mathbb{E}[Y] + \mu_X\mu_Y = \mathbb{E}[XY] - \mu_X\mu_Y$.

- $\operatorname{Cov} > 0$: tend to be above/means together — positive association.
- $\operatorname{Cov} < 0$: opposite directions.
- $\operatorname{Cov} = 0$: **uncorrelated** — no *linear* association.
- $\operatorname{Cov}(X, X) = \operatorname{Var}(X)$.

**Worked example — a small joint table.** $X, Y \in \{0,1\}$ with joint probabilities:

|  | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | 0.3 | 0.2 |
| $X=1$ | 0.2 | 0.3 |

Marginals: $P(X{=}0) = 0.5$, $P(X{=}1) = 0.5$; same for $Y$. So $\mathbb{E}[X] = \mathbb{E}[Y] = 0.5$.

$$
\mathbb{E}[XY] = 0\cdot0\cdot0.3 + 0\cdot1\cdot0.2 + 1\cdot0\cdot0.2 + 1\cdot1\cdot0.3 = 0.3
$$

$$
\operatorname{Cov}(X,Y) = 0.3 - (0.5)(0.5) = 0.05 > 0 \quad\text{(positive association)}
$$

## Correlation

Covariance's magnitude depends on units — rescale a variable and the covariance rescales. **Pearson correlation** normalizes:

$$
\rho = \frac{\operatorname{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

$\rho = \pm 1$ exactly when one variable is a perfect linear function of the other; $\rho = 0$ means uncorrelated. (Dependence without correlation exists: $Y = X^2$ with symmetric $X$ gives $\operatorname{Cov}(X, X^2) = \mathbb{E}[X^3] - 0 = 0$, yet $Y$ is completely determined by $X$.)

## The Covariance Matrix

For a vector random variable $\mathbf{X} = (X_1, \ldots, X_n)^\top$, collect pairwise covariances:

$$
\Sigma = \operatorname{Cov}(\mathbf{X}) =
\begin{bmatrix}
\operatorname{Var}(X_1) & \operatorname{Cov}(X_1,X_2) & \cdots \\
\operatorname{Cov}(X_2,X_1) & \operatorname{Var}(X_2) & \cdots \\
\vdots & & \ddots
\end{bmatrix}
$$

Properties: **symmetric** ($\operatorname{Cov}(X_i, X_j) = \operatorname{Cov}(X_j, X_i)$) and **positive semi-definite** ($\mathbf{v}^\top\Sigma\mathbf{v} = \operatorname{Var}(\mathbf{v}^\top\mathbf{X}) \ge 0$ — variance cannot be negative). Both are section 01 concepts; $\Sigma$ is the canonical example of a symmetric positive-semidefinite matrix, and its eigenvectors are the principal components of section 07.

Diagonal entries = variances; off-diagonal = covariances. If all off-diagonal are zero, the components are pairwise uncorrelated (for Gaussians: independent — but not in general).

## Connection to AI

- **Expected loss.** The objective in supervised learning *is* an expectation: $J = \mathbb{E}_{(x,y)\sim\text{data}}[L(f(x), y)]$, estimated by the average over the training batch. Minimizing the empirical average approximates minimizing the true expectation.
- **Variance = uncertainty.** Model confidence relates to predictive variance; ensemble methods exist to *estimate* it. High-variance estimates across data resamples signal instability (bias–variance, section 04).
- **Covariance of data = structure.** PCA diagonalizes the covariance matrix (section 07); whitening (dividing by $\sqrt{\text{variance}}$) normalizes features before similarity computations.
- **Gradient noise.** Minibatch gradients are random variables whose covariance across batches determines SGD's noisy trajectory — the same math that lets noise escape saddles.
- **Regularization as variance control.** Constraining weights limits how wildly predictions vary — bias–variance trade-off, section 04/05.

## Common Mistakes

- Assuming $\mathbb{E}[g(X)] = g(\mathbb{E}[X])$ — false for non-linear $g$ (this is why variance is computed as $\mathbb{E}[X^2] - \mu^2$ and not as $\mathbb{E}[|X-\mu|]$ shorthands).
- Computing $\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ and concluding independence from a non-zero value — non-zero covariance shows *dependence*; zero covariance does not show independence.
- Forgetting the square when scaling variance: $\operatorname{Var}(2X) = 4\operatorname{Var}(X)$.
- Confusing covariance matrix (symmetric, PSD, of variables) with correlation matrix (same shape, diagonal 1, entries in $[-1,1]$).
- Using population formulas ($1/n$) on samples when an unbiased estimate ($1/(n-1)$) is wanted — section 04 addresses estimators.

## Exercises

1. Compute $\mathbb{E}[X]$ for $p(1) = 0.2$, $p(2) = 0.5$, $p(3) = 0.3$.
2. Compute $\operatorname{Var}(X)$ for the same distribution (via $\mathbb{E}[X^2] - \mu^2$).
3. A Bernoulli trial with $p = 0.7$: give $\mathbb{E}[X]$ and $\operatorname{Var}(X)$.
4. Given $\mathbb{E}[X] = 2$, $\mathbb{E}[Y] = 5$, $\mathbb{E}[XY] = 11$: compute $\operatorname{Cov}(X,Y)$.
5. For the joint table above, compute $\operatorname{Var}(X)$, $\operatorname{Var}(Y)$, and the correlation $\rho$.
6. Show that $\operatorname{Cov}(aX + b, Y) = a\operatorname{Cov}(X, Y)$ using the definition $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ and linearity.
7. If $X$ and $Y$ are independent, show $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ *using the joint-PMF sum* (this direction needs independence; state where).
