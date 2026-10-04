# Hypothesis Tests and Confidence Intervals

## Why It Matters for AI

"Model A scored 94.2%, model B scored 93.8% — is A better?" Without uncertainty quantification, that question cannot be answered: test sets are samples, and sample differences fluctuate. Hypothesis tests and confidence intervals are the elementary tools for deciding whether an observed difference is signal or noise. This chapter covers the basics honestly, including what these tools do *not* say.

## Setup: A Hypothesis Test

A **null hypothesis** $H_0$ is the claim of "no effect / no difference" — the default you need evidence to reject. The **alternative** $H_1$ is what you would conclude if the data are surprising under $H_0$.

Example: $H_0$: the coin is fair ($p = 0.5$); $H_1$: it is not ($p \ne 0.5$).

**Procedure:**

1. Assume $H_0$ true.
2. Compute a **test statistic** $T$ — a number summarizing the data, whose distribution is known under $H_0$.
3. Find how surprising $T$'s value is: the **p-value**.
4. If the p-value is below a threshold $\alpha$ (significance level, conventionally $0.05$), reject $H_0$; otherwise, fail to reject.

## The p-value

$$
p\text{-value} = P(\text{data at least as extreme as observed} \mid H_0 \text{ is true})
$$

It is *not* $P(H_0 \mid \text{data})$ — the probability of the hypothesis given the data (that would be Bayes' theorem with a prior over hypotheses). The p-value conditions on $H_0$ and asks about the data.

**Worked example — is this coin fair?** 100 flips, 60 heads. Under $H_0$, heads count $X \sim \text{Binomial}(100, 0.5)$:

$$
\mathbb{E}[X] = 50, \qquad \operatorname{SD}(X) = \sqrt{100 \cdot 0.5 \cdot 0.5} = 5
$$

(observed 60 is 2 standard deviations above the mean). $z$-statistic:

$$
z = \frac{0.6 - 0.5}{\sqrt{0.5 \cdot 0.5 / 100}} = \frac{0.1}{0.05} = 2
$$

For large samples, under $H_0$ this statistic is approximately standard normal. Two-sided p-value:

$$
P(|Z| \ge 2) = 2\,(1 - \Phi(2)) \approx 2 \times 0.0228 = 0.0455
$$

$0.0455 < 0.05$ → reject $H_0$: the data are (just) surprising enough under fairness. Not overwhelming evidence — and "reject at 5%" is a statement about the procedure, not certainty about the coin.

**Interpretation rules:**

- Small p-value: data would be unusual *if* $H_0$ were true.
- Large p-value: data are *compatible* with $H_0$ — **not** proof that $H_0$ is true (absence of evidence ≠ evidence of absence).
- The threshold $\alpha$ is a chosen loss trade-off (false alarms vs missed effects), not a law of nature.

## Confidence Intervals

A **confidence interval** estimates an unknown parameter with a range instead of a point. For a mean with known standard deviation $\sigma$:

$$
\bar{x} \;\pm\; z^* \cdot \frac{\sigma}{\sqrt{n}}
$$

$\dfrac{\sigma}{\sqrt n}$ is the **standard error** (the $1/\sqrt n$ sampling spread of $\bar X$ from section 02); $z^*$ is the normal quantile: $z^* = 1.96$ for 95% (the value with $95\%$ of the normal mass inside $\pm z^*$).

**Worked example.** Sample of $n = 25$ measurements: $\bar x = 52$, known $\sigma = 10$.

$$
\text{SE} = \frac{10}{\sqrt{25}} = 2, \qquad
95\%\ \text{CI} = 52 \pm 1.96 \times 2 = 52 \pm 3.92 = [48.08,\ 55.92]
$$

**Honest interpretation of "95%":** the *procedure* — "take a sample, form $\bar x \pm 1.96\,\text{SE}$" — captures the true $\mu$ in $95\%$ of repetitions of the whole experiment. It is a property of the method's long-run behavior, not a single fixed interval's probability of containing $\mu$ (the parameter is either in this interval or not).

When $\sigma$ is unknown and $n$ small, use the sample standard deviation $s$ and the $t$-distribution (slightly wider than normal) — for large $n$ the difference vanishes; not developed here.

**Relation between tests and intervals:** 95% CI excludes $\mu_0$ ⟺ the 5% test of $H_0: \mu = \mu_0$ rejects — two views of the same calculation.

## What Both Tools Assume (and Where They Break)

- **Random sample** (or randomness from the assignment in experiments) — correlated or shifted data invalidate the standard errors.
- **Independence across observations** — violated by repeated users, time series, clustered data; effective sample size is then smaller than $n$, intervals too narrow.
- **Model form** (normality for small-sample exactness; approximations otherwise).
- **Fixed analysis plan** — checking 20 comparisons and reporting the "significant" one inflates false positives enormously (**multiple comparisons**; corrections or held-out confirmation are required).

## Connection to AI

- **Comparing models.** Two classifiers on one test set produce paired outcomes; a McNemar test (the right tool for paired binary results) or bootstrap intervals answer "is the difference real?". Quoting score differences without uncertainty is unscientific when gaps are fractions of a point.
- **A/B tests** at scale are hypothesis tests: did the new model improve the metric, or did noise? The $n$-sized sample logic of this chapter sets how long to run the test.
- **Training-run variance.** Same model, different seeds → different scores; reporting best-run results without spread overstates performance. Mean ± std over seeds is a mini confidence statement.
- **Evaluation leakage** is the multiple-comparisons problem in ML clothing: tuning on the test set makes its score a biased estimate (section 02's optimism bias).
- **What tests do not give:** a probability that your model is correct. For that you would need priors — chapter 04's Bayesian direction.

## Common Mistakes

- Reading the p-value as $P(H_0 \mid \text{data})$ — it is $P(\text{extreme data} \mid H_0)$.
- "Fail to reject" ≠ "accept" — no evidence against $H_0$ is not evidence for it.
- Interpreting a 95% CI as "95% probability $\mu$ is in *this* interval" — the long-run reading is over repeated samples.
- Testing many hypotheses without correction — expected false positives grow with the number of tests ($\alpha \times$ number of tests under all-null).
- Neglecting dependence: $n = 10{,}000$ clicks from 100 users is not $10{,}000$ independent observations.
- Comparing single-run scores from different random seeds and declaring a winner from noise.

## Exercises

1. State $H_0$ and $H_1$ for: "does the new model's accuracy exceed 90%?"
2. Compute the $z$-statistic and approximate two-sided p-value for: $n = 100$, sample proportion $0.57$ under $H_0: p = 0.5$. (Use the chapter's standard-error formula; find $P(|Z| \ge z)$ approximately for your $z$.)
3. 95% CI for a mean: $\bar x = 100$, $\sigma = 15$, $n = 9$. Compute it (SE first). ($z^* = 1.96$.)
4. True or false, with a one-line reason: "A p-value of 0.03 means there is a 97% chance the alternative hypothesis is true."
5. If you run 100 independent tests at $\alpha = 0.05$ when all nulls are true, how many rejections do you expect by chance?
6. A model scores 94.2% vs 93.8% on the *same* test set; a colleague wants to claim superiority from one run. Name two statistical problems with that claim.
