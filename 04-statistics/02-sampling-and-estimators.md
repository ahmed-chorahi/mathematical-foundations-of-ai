# Sampling and Estimators

## Why It Matters for AI

You never see the true data distribution — you see a sample. Every quantity a model learns is an **estimate** computed from finite data, and estimates have systematic error (**bias**) and random error (**variance**). Understanding what sampling does to an estimate is the foundation of the bias–variance trade-off, of why validation sets are held out, and of why "the training loss" is not "the true loss".

## Random Sampling

A **random sample** $X_1, \ldots, X_n$ is drawn so that each draw follows the population distribution $P$ and draws are **independent** (i.i.d. assumption: independent and identically distributed).

Why the assumption matters: it makes sample statistics behave predictably — the mean of $n$ draws concentrates around $\mu$ with spread $\sigma/\sqrt{n}$ (used below and in chapter 05). Real data violate it in familiar ways (time series correlate, users contribute multiple rows, distributions shift over time); when they do, standard formulas overstate certainty.

## Estimators

An **estimator** $\hat{\theta}$ is any function of the sample that produces a guess for a parameter:

$$
\hat{\theta} = g(X_1, \ldots, X_n)
$$

Its computed value on your data is an **estimate** (the recipe versus the number).

Examples: $\bar X$ estimates $\mu$; $s^2$ estimates $\sigma^2$; the fraction of successes estimates $p$.

Each estimator has a **sampling distribution** — the distribution of $\hat\theta$ across possible samples. (Random variable first, number second: chapter 03's distinction applied to statistics.)

## Bias

$$
\operatorname{Bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta
$$

- **Unbiased:** $\mathbb{E}[\hat\theta] = \theta$ — right *on average* over samples.
- **Biased:** systematically off.

**Worked example 1 — sample mean is unbiased.** With $X_i$ i.i.d., mean $\mu$:

$$
\mathbb{E}[\bar X] = \mathbb{E}\left[\frac{1}{n}\sum X_i\right] = \frac{1}{n}\sum \mathbb{E}[X_i] = \frac{1}{n}\cdot n\mu = \mu
$$

(linearity of expectation, section 03 — no independence needed for *this* step).

**Worked example 2 — dividing by $n$ biases variance low.** Define $\tilde S^2 = \frac1n\sum(X_i - \bar X)^2$. We compute $\mathbb{E}[\tilde S^2]$ using two facts:

1. $\operatorname{Var}(\bar X) = \frac{\sigma^2}{n}$: $\operatorname{Var}(\bar X) = \operatorname{Var}(\frac1n \sum X_i) = \frac{1}{n^2}\sum\operatorname{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}$ (independence kills cross terms).
2. The identity $\sum_i (X_i - \bar X)^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2$ (expand and simplify — the cross term $2(\bar X - \mu)\sum(X_i - \bar X)$ vanishes because deviations from the mean sum to zero).

Take expectations of (2) divided by $n$:

$$
\mathbb{E}[\tilde S^2] = \frac{1}{n}\left(n\sigma^2 - n\cdot\frac{\sigma^2}{n}\right) = \sigma^2 - \frac{\sigma^2}{n} = \sigma^2\cdot\frac{n-1}{n}
$$

Short by factor $(n-1)/n$ — **biased low**. Multiplying by $\frac{n}{n-1}$ gives $s^2 = \frac{1}{n-1}\sum(X_i-\bar X)^2$ with expectation exactly $\sigma^2$: **unbiased**, the formula from chapter 01. The $n-1$ correction is now derived, not folklore.

## Variance of an Estimator

Bias is not the only error — the estimator also *varies* across samples:

$$
\operatorname{Var}(\hat\theta) = \mathbb{E}\left[(\hat\theta - \mathbb{E}[\hat\theta])^2\right]
$$

The sample mean has $\operatorname{Var}(\bar X) = \sigma^2 / n$ — variance shrinks as $1/n$: four times the data, half the standard deviation of the estimate ($\sqrt{1/4} = 1/2$). This rate $\sigma/\sqrt{n}$ is the **standard error** and reappears in chapter 05's intervals.

## Mean Squared Error

Combine both errors:

$$
\operatorname{MSE}(\hat\theta) = \mathbb{E}\left[(\hat\theta - \theta)^2\right]
= \operatorname{Bias}(\hat\theta)^2 + \operatorname{Var}(\hat\theta)
$$

**Derivation** (short): write $\hat\theta - \theta = (\hat\theta - \mathbb{E}[\hat\theta]) + (\mathbb{E}[\hat\theta] - \theta)$ — noise around the estimator's mean, plus the offset of that mean. Square: cross term's expectation is $\operatorname{Cov}(\hat\theta - \mathbb{E}\hat\theta, \text{constant}) = 0$, leaving bias² + variance.

The **bias–variance trade-off**: constraining an estimator introduces bias but can cut variance by more than the bias costs, lowering total MSE. Extreme case: the constant estimator $\hat\theta = 0$ has zero variance and MSE $\theta^2$ — good if $\theta$ is near 0, terrible otherwise. Between it and the unbiased $\bar X$ sits a **shrinkage** estimator $\bar X \cdot c$ (some $c < 1$): biased, but often with smaller MSE than $\bar X$ itself. This trade-off *is* why regularization (bias for variance reduction) can improve test performance — chapter 04 and section 05 make the connection concrete for models.

## Sample Size

The estimator's standard deviation shrinks like $1/\sqrt n$ — meaning **diminishing returns**: to halve the error you must quadruple the data. $n$ needed for error $e$: roughly $n \approx (z\,\sigma/e)^2$ (the interval formula of chapter 05 inverted).

## Connection to AI

- **Training is estimation from a sample.** Model parameters estimate the function that generated the data; the generalization gap (train vs test error) is estimation error, decomposed by this chapter's tools.
- **Bias–variance for models.** Underfitting = high bias; overfitting = high variance (informally; the model version is stated precisely in section 07's orbit). Regularization trades bias for variance.
- **Validation sets are held-out samples** used to estimate generalization error — their size follows the $1/\sqrt{n}$ logic: small validation sets give noisy comparisons (chapter 05 returns).
- **Averages over batches** estimate the true gradient; minibatch noise is the estimator's variance (section 05).
- **Learning curves** (error vs $n$) show the $1/\sqrt{n}$-style decay of estimation error — and its floor, the model's bias.

## Common Mistakes

- Treating the sample distribution as the population — quotes like "the data has mean 5" mean *the sample* has mean 5.
- Assuming unbiasedness makes an estimator good — unbiased with huge variance can lose to a biased estimator with small MSE.
- Forgetting that *any* model-selection step (picking the best of 10 configurations on validation) uses the same sample twice — the selected estimate is optimistically biased.
- Believing more data fixes bias — bias persists as $n \to \infty$; only variance shrinks.
- Using $\operatorname{Var}(\bar X) = \sigma^2$ (wrong — that is for one draw) instead of $\sigma^2/n$.

## Exercises

1. Show that $\bar X$ is unbiased for $\mu$ when $\mathbb{E}[X_i] = \mu$ (write the steps using linearity).
2. Compute $\operatorname{Bias}(\tilde S^2)$ for $\tilde S^2 = \frac1n\sum(X_i - \bar X)^2$ — use the result derived above.
3. The standard deviation of a single draw is $\sigma = 6$. Give the standard error of the mean for $n = 9$, and for $n = 36$.
4. Decompose MSE for an estimator with bias $3$ and variance $16$.
5. An estimator has $\mathbb{E}[\hat\theta] = \theta$ but $\operatorname{Var}(\hat\theta) = 1000$. Is it unbiased? Is it good? Explain both.
6. State what happens to the standard error if you multiply the sample size by 4.
