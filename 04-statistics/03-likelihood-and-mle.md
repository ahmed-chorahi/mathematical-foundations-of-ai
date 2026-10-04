# Likelihood and Maximum Likelihood Estimation

## Why It Matters for AI

Ask: *which parameters best explain the data I observed?* The answer — maximize the **likelihood** — is the most common estimation principle in machine learning, and its negative logarithm *is* the loss function for most models you have heard of (cross-entropy, Gaussian squared error). This chapter derives that connection from the definition.

## Likelihood

Given a model with parameter $\theta$ (say, a Bernoulli success probability $p$) and observed data $D = x_1, \ldots, x_n$:

- **Probability of data given parameters:** $P(D \mid \theta)$ — a function of the data with $\theta$ *fixed*.
- **Likelihood:** $L(\theta) = P(D \mid \theta)$ — the same number viewed as a function of $\theta$ with the data *fixed*.

Same expression, different argument. Probability asks "how likely is this data?" (θ known); likelihood asks "how well does this θ fit this data?" (data known).

**Worked example — coin.** Model: Bernoulli($p$), $p$ unknown. Observe 3 heads in 5 flips ($D = \{H, H, H, T, T\}$):

$$
L(p) = P(D \mid p) = p^3 (1-p)^2
$$

(each head contributes factor $p$, each tail $1-p$ — independence of flips).

Values: $L(0.5) = 0.125 \cdot 0.25 = 0.03125$; $L(0.6) = 0.216 \cdot 0.16 = 0.03456$; $L(0.9) = 0.729 \cdot 0.01 = 0.00729$. The fit is best somewhere near $0.6$ — maximum likelihood finds it exactly.

## Log-Likelihood

Maximizing $L(\theta)$ = maximizing $\log L(\theta)$ (logarithm is increasing — same argmax):

$$
\ell(\theta) = \log L(\theta)
$$

Why bother: products become sums, powers become multipliers.

**Coin example:**

$$
\ell(p) = 3\ln p + 2\ln(1-p)
$$

Sum-of-logs is far easier to differentiate than $p^3(1-p)^2$, and numerically stable (no underflow from multiplying thousands of small probabilities — practical computers multiply probabilities like $10^{-400}$ down to zero; logs stay finite).

## Maximum Likelihood Estimation (MLE)

$$
\boxed{\;\hat\theta_{\text{MLE}} = \arg\max_\theta \; \log P(D \mid \theta)\;}
$$

Procedure: write $P(D \mid \theta)$ (assume the model), take log, differentiate, set to zero, solve.

**Worked example — the coin, fully.** With $k$ heads in $n$ flips:

$$
\ell(p) = k \ln p + (n-k)\ln(1-p)
$$

$$
\ell'(p) = \frac{k}{p} - \frac{n-k}{1-p}
\;\stackrel{!}{=}\; 0
$$

$$
\frac{k}{p} = \frac{n-k}{1-p} \;\Rightarrow\; k(1-p) = (n-k)p \;\Rightarrow\; k = np \;\Rightarrow\;
\boxed{\hat p = \frac{k}{n}}
$$

The observed proportion! ($3/5 = 0.6$ for our data — matching the numerical scan above.) Second derivative:

$$
\ell''(p) = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0
$$

for all $p \in (0,1)$ — the log-likelihood is concave, so the stationary point is the global maximum. (Concavity checking is the likelihood version of section 07's second-derivative test.)

**Worked example — Gaussian mean.** $X_i \sim N(\mu, \sigma^2)$, $\sigma^2$ known, data $x_1, \ldots, x_n$:

$$
\ell(\mu) = \sum_{i=1}^n \left[-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2}\right]
$$

Differentiate in $\mu$ (constants drop):

$$
\ell'(\mu) = \sum_{i=1}^n \frac{(x_i - \mu)}{\sigma^2} = \frac{1}{\sigma^2}\left(\sum x_i - n\mu\right) \stackrel{!}{=} 0
\;\Rightarrow\; \hat\mu = \frac{1}{n}\sum_i x_i = \bar x
$$

The MLE of the mean is the sample mean — the estimator from chapter 01, now derived from a principle.

For $\sigma^2$ (treating it as the variable, $\mu$ known or plugged in): maximizing gives $\hat\sigma^2_{\text{MLE}} = \frac1n\sum(x_i - \bar x)^2$ — the **biased** $1/n$ version. Maximum likelihood does not promise unbiasedness; the two estimates answer slightly different questions (exact fit to data vs. unbiased estimation).

## The Pattern: Loss = Negative Log-Likelihood

Maximizing $\ell(\theta)$ = minimizing $-\ell(\theta)$. Define the loss as the negative log-likelihood and training is *always* the same operation:

| model | likelihood assumption | $-\log L$ per example |
|---|---|---|
| Bernoulli classification | target $\in \{0,1\}$, $P(1) = \hat p$ | $-[y\ln\hat p + (1-y)\ln(1-\hat p)]$ |
| Gaussian noise | $y = f(x) + \varepsilon,\ \varepsilon \sim N(0,\sigma^2)$ | $\propto (y - f(x))^2$ |
| categorical (softmax) | $P(\text{true class}) = \hat p_{\text{true}}$ | $-\ln \hat p_{\text{true}}$ |

Read row 2 carefully: **squared-error loss is a log-likelihood** — assuming Gaussian noise makes MSE the (negative) log-likelihood. Row 3 is cross-entropy (section 06) — same construction for categorical outputs. The two most common losses in machine learning are maximum-likelihood solutions under Gaussian and categorical assumptions. "Why this loss?" is answered by "which likelihood?"

**Worked example — cross-entropy from likelihood.** One example, softmax output $\hat p$ over classes, true class $y$:

$$
L(\theta) = \hat p_y = \operatorname{softmax}(\text{logits})_y,
\qquad
-\log L = -\ln \hat p_y
$$

Maximizing the probability assigned to the truth = minimizing negative log-likelihood. Section 08 uses this loss for classification networks; section 10 for language models.

## Properties Worth Knowing

- **Consistency:** under standard conditions, $\hat\theta_{\text{MLE}} \to \theta$ (the truth) as $n \to \infty$.
- **Invariance:** if $\hat\theta$ maximizes $L$, then $g(\hat\theta)$ is the MLE of $g(\theta)$ — no re-derivation needed for transformations.
- **Asymptotic efficiency:** MLE achieves the lowest possible variance among unbiased estimators in the large-sample limit (Cramér–Rao bound — stated, not proved).
- **Misspecification:** if the true data-generating distribution is outside your model family, MLE converges to the *closest* family member (in KL divergence — section 06 gives the precise meaning of "closest").

## Connection to AI

- **Training = maximum likelihood.** Gradient descent on cross-entropy or MSE is gradient ascent on the log-likelihood — the standard account of what supervised training *is*.
- **MLE explains the losses you already know**: MSE (Gaussian), cross-entropy (categorical/Bernoulli), diffusion objectives (Gaussian with learned variance) — all $-\log P(D \mid \theta)$.
- **Overfitting viewed through likelihood:** a flexible model can assign near-1 likelihood to training data by memorization — likelihood on training data alone does not protect against overfitting (held-out likelihood does; priors in the next chapter do).
- **Perplexity** (section 10) is the exponentiated average negative log-likelihood per token — a likelihood measure directly comparable across models.

## Common Mistakes

- Confusing $P(\theta \mid D)$ with $P(D \mid \theta)$ — the first is posterior (needs Bayes and a prior), the second is likelihood. MLE uses only the second.
- Maximizing $L$ directly when $\prod$ underflows — use $\log$; the argmax agrees, the numerics do not.
- Assuming MLE is unbiased — the $\hat\sigma^2$ case above shows it need not be.
- Dropping the Jacobian when transforming densities (MLE's invariance handles parameter transformations, but density transformations are a different matter).
- Declaring MLE "correct" without checking the likelihood family — the assumptions are part of the result.

## Exercises

1. Write $L(p)$ and $\ell(p)$ for a Bernoulli model with data $\{1, 1, 0, 1\}$ (4 flips).
2. Maximize $\ell(p)$ from problem 1 — verify the general formula $\hat p = k/n$.
3. Two flips, both heads: give $\hat p$; then say what $L(0.5)$ and $L(\hat p)$ are, and which is larger.
4. For $X_i \sim N(\mu, 1)$ with data $\{2, 4, 6\}$, derive $\hat\mu$ by differentiating $\ell(\mu)$ and compute it.
5. Show that for Gaussian noise with unknown $\sigma^2$, differentiating $\ell(\sigma^2)$ (with $\mu$ known) gives $\hat\sigma^2 = \frac1n\sum(x_i - \mu)^2$.
6. State why $\arg\max \log L = \arg\max L$ and one practical advantage of using the log.
