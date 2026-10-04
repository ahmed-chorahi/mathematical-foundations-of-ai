# 04 — Statistics: Solutions

## 01-descriptive-statistics.md

**1. Mean and median of $[3, 1, 4, 1, 5, 9, 2, 6]$.**
Sum $= 31$, $n = 8$: $\bar x = 31/8 = 3.875$.
Sorted: $[1, 1, 2, 3, 4, 5, 6, 9]$; middle two are $3$ and $4$: median $= 3.5$.

**2. Sample variance of $[1, 3, 5]$.**
$\bar x = 3$; deviations $-2, 0, 2$; squared: $4, 0, 4$; sum $= 8$.
$$
s^2 = \frac{8}{n-1} = \frac{8}{2} = 4 \quad(\text{sample});\qquad
\frac{8}{n} = \frac{8}{3} \approx 2.67 \quad(\text{MLE version})
$$

**3. Z-score of 14 with mean 10, std 2.**
$$
z = \frac{14 - 10}{2} = 2
$$
Two standard deviations above the mean.

**4. Correlation sign of $x = [1,2,3]$, $y = [3,2,1]$.**
$y = 4 - x$ — perfectly decreasing: $r = -1$ (negative sign). (Any $x$ up pairs with $y$ down → negative linear relationship.)

**5. Why $n-1$ corrects a bias (one sentence).**
Deviations are measured from $\bar x$ (fitted to the same data), which makes them systematically smaller than deviations from the true mean $\mu$; dividing by $n-1$ rescales so the expectation equals $\sigma^2$ exactly (the $n/(n-1)$ inflation offsets the $1 - 1/n$ shrinkage).

**6. Covariance matrix $\begin{bmatrix}4&0\\0&9\end{bmatrix}$.**
Zero off-diagonal: the two features are uncorrelated (no linear association). Variances are $4$ and $9$, so standard deviations are $\sqrt 4 = 2$ and $\sqrt 9 = 3$.

## 02-sampling-and-estimators.md

**1. Unbiasedness of $\bar X$.**
$$
\mathbb{E}[\bar X] = \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n X_i\right]
= \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i]
= \frac{1}{n}\cdot n\mu = \mu
$$
by linearity of expectation (independence is not needed here — only equal means $\mathbb{E}[X_i] = \mu$).

**2. Bias of $\tilde S^2 = \frac1n\sum(X_i - \bar X)^2$.**
From the derivation: $\mathbb{E}[\tilde S^2] = \sigma^2\frac{n-1}{n}$, so
$$
\operatorname{Bias} = \sigma^2\frac{n-1}{n} - \sigma^2 = -\frac{\sigma^2}{n}
$$
Negative — systematically too small by $\sigma^2/n$.

**3. Standard error of the mean, $\sigma = 6$, $n = 9$ and $n = 36$.**
$$
\text{SE} = \frac{\sigma}{\sqrt n}:\quad
n = 9 \Rightarrow \frac{6}{3} = 2;\qquad
n = 36 \Rightarrow \frac{6}{6} = 1
$$

**4. MSE with bias 3, variance 16.**
$$
\operatorname{MSE} = 3^2 + 16 = 25
$$

**5. Unbiased but variance 1000?**
Unbiased: yes, by definition ($\mathbb{E}[\hat\theta] = \theta$). Good: not necessarily — $\operatorname{MSE} = 0 + 1000 = 1000$; a biased estimator with total MSE below 1000 beats it. Unbiasedness alone does not measure quality.

**6. Sample size ×4.**
SE scales as $1/\sqrt n$: $\frac{1}{\sqrt{4}} = \frac12$ — the standard error is **halved**.

## 03-likelihood-and-mle.md

**1. $L(p)$ and $\ell(p)$ for data $\{1,1,0,1\}$.**
Three 1s and one 0:
$$
L(p) = p^3(1-p), \qquad \ell(p) = 3\ln p + \ln(1-p)
$$

**2. Maximize; verify $\hat p = k/n$.**
$$
\ell'(p) = \frac{3}{p} - \frac{1}{1-p} = 0
\;\Rightarrow\; 3(1-p) = p \;\Rightarrow\; 3 = 4p \;\Rightarrow\; \hat p = 0.75 = \frac{3}{4} = \frac{k}{n} \;✓
$$
$\ell''(p) = -3/p^2 - 1/(1-p)^2 < 0$ → maximum ✓

**3. Two flips, both heads.**
$\hat p = 2/2 = 1$ (MLE sits on the boundary).
$L(0.5) = 0.5^2 = 0.25$; $L(1) = 1^2 = 1$. $L(\hat p) = 1 > 0.25$ ✓ — by construction the MLE maximizes the likelihood (though $\hat p = 1$ is a reminder that MLE with tiny samples can be extreme).

**4. $\hat\mu$ for $\{2,4,6\}$, $N(\mu,1)$.**
$$
\ell(\mu) = -\frac{1}{2}\sum(x_i - \mu)^2 + \text{const}, \quad
\ell'(\mu) = \sum(x_i - \mu) = 0
\;\Rightarrow\; \hat\mu = \bar x = \frac{2+4+6}{3} = 4
$$

**5. MLE of $\sigma^2$, $\mu$ known.**
$$
\ell(\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2
$$
Let $v = \sigma^2$. $\dfrac{d\ell}{dv} = -\dfrac{n}{2v} + \dfrac{\sum(x_i-\mu)^2}{2v^2} = 0 \Rightarrow n = \dfrac{\sum(x_i-\mu)^2}{v} \Rightarrow \hat\sigma^2 = \dfrac1n\sum(x_i-\mu)^2$.
(Second derivative $n/(2v^2) - \sum(\cdot)/v^3$ evaluated at the solution is negative — a maximum.)

**6. Why log, and its advantage.**
$\log$ is strictly increasing, so it preserves the argmax: $\arg\max L = \arg\max \log L$ (both attained at the same $\theta$). Advantages: products of likelihoods become sums (easier differentiation, factorized gradients), powers come down as multipliers, and numerically logs avoid underflow when many small probabilities are multiplied.

## 04-map-estimation.md

**1. MAP objective.**
$$
\hat\theta_{\text{MAP}} = \arg\max_\theta \Big[\log P(D \mid \theta) + \log P(\theta)\Big]
$$
($P(D)$ dropped: constant in $\theta$.)

**2. Uniform prior on $p$.**
$P(p) = 1$ on $(0,1)$, so $\log P(p) = 0$ (constant): the log-posterior equals the log-likelihood and $\hat p_{\text{MAP}} = k/n = 3/5 = 0.6$ — the same as MLE. A uniform prior *is* the statement "no preference", so it cannot move the mode.

**3. Prior $\propto p(1-p)$.**
$$
\log \text{post}(p) = 3\ln p + 2\ln(1-p) + \ln p + \ln(1-p) + c = 4\ln p + 3\ln(1-p) + c
$$
$$
\frac{d}{dp} = \frac{4}{p} - \frac{3}{1-p} = 0 \;\Rightarrow\; 4 - 4p = 3p \;\Rightarrow\; \hat p = \frac{4}{7} \approx 0.571
$$
$0.5 < 0.571 < 0.6$ ✓ — shrunk from the MLE (0.6) toward the prior center (0.5), exactly as expected: the prior pulls where data are scarce (only 5 flips).

**4. $\min_\theta (\theta-10)^2 + \lambda\theta^2$.**
$$
2(\theta - 10) + 2\lambda\theta = 0 \;\Rightarrow\; \theta = \frac{10}{1+\lambda}
$$
$\lambda = 0 \Rightarrow 10$ (pure data); $\lambda = 1 \Rightarrow 5$; $\lambda = 9 \Rightarrow 1$ (prior dominates).

**5. $L_1$ objective and its prior.**
$$
\min_\theta\; -\log P(D \mid \theta) + \frac{1}{b}\|\theta\|_1
$$
from a **Laplace** prior $\theta_j \sim \text{Laplace}(0, b)$ (density $\propto e^{-|\theta_j|/b}$).

**6. $\tau \to \infty$ recovers MLE.**
The penalty $\frac{1}{2\tau^2}\|\theta\|^2 \to 0$, so the posterior objective becomes the plain log-likelihood — an infinitely wide Gaussian prior imposes no preference, and MAP = MLE.

## 05-tests-and-intervals.md

**1. $H_0$, $H_1$ for "does accuracy exceed 90%?".**
$H_0: p \le 0.90$ (accuracy is at most 90% — the no-improvement claim), $H_1: p > 0.90$ (one-sided alternative).

**2. $z$ and p-value for $n = 100$, $\hat p = 0.57$, $H_0: p = 0.5$.**
$$
\text{SE} = \sqrt{\frac{0.5 \times 0.5}{100}} = 0.05,\qquad
z = \frac{0.57 - 0.5}{0.05} = 1.4
$$
$$
p\text{-value} = 2\,[1 - \Phi(1.4)] \approx 2 \times 0.0808 = 0.1616 \approx 0.16
$$
$0.16 > 0.05$ → do not reject: results this far from 0.5 are common by chance.

**3. 95% CI, $\bar x = 100$, $\sigma = 15$, $n = 9$.**
$$
\text{SE} = \frac{15}{3} = 5;\qquad 100 \pm 1.96 \times 5 = 100 \pm 9.8 = [90.2,\ 109.8]
$$

**4. "p = 0.03 means 97% chance $H_1$ is true."**
**False.** The p-value is $P(\text{data this extreme} \mid H_0)$ — a statement about data assuming the null, not a probability about hypotheses; converting between them requires priors (Bayes).

**5. 100 tests at $\alpha = 0.05$, all nulls true.**
$100 \times 0.05 = 5$ expected false rejections by chance alone — the reason uncorrected multiple testing manufactures "discoveries".

**6. Two problems with claiming superiority from 94.2% vs 93.8% in one run.**
1. **Sampling noise:** one test set gives one realization; the difference (0.4 points) may be within the sampling standard error of the comparison — no uncertainty was quantified (also run-to-run seed variance).
2. **Wrong comparison setup:** the two models' predictions on the same set are *paired* — a paired test (e.g., McNemar) is needed; if the test set was used during development, the estimate is additionally biased (leakage/multiple comparisons).
