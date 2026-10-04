# 04 — Statistics

Statistics answers a question the previous sections assume away: *you have data, not the true distribution — what can you honestly conclude?* This section covers the statistical mathematics that AI actually uses: summarizing data, understanding estimators, maximum likelihood, MAP estimation (the origin of regularization), and the basics of testing and intervals. It stays deliberately narrower than a full statistics course.

## Contents

| File | Topic | AI application |
|---|---|---|
| [01-descriptive-statistics.md](01-descriptive-statistics.md) | mean, median, variance, std, covariance, correlation; population vs sample | data summaries, normalization |
| [02-sampling-and-estimators.md](02-sampling-and-estimators.md) | sampling, estimators, bias, variance of estimators, MSE | what training on data can and cannot deliver |
| [03-likelihood-and-mle.md](03-likelihood-and-mle.md) | likelihood, log-likelihood, maximum likelihood estimation | loss functions as negative log-likelihoods |
| [04-map-estimation.md](04-map-estimation.md) | priors, MAP, Gaussian/Laplace priors → L2/L1 penalties | regularization as Bayesian inference |
| [05-tests-and-intervals.md](05-tests-and-intervals.md) | hypothesis tests, p-values, confidence intervals | comparing models honestly, evaluation |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Section 01 (vectors, covariance as matrix structure), section 03 (expectation, Bayes' theorem, distributions).

## Through-Line

```text
Data → summaries (01)
     → sampling limitations (02)
     → likelihood: fit by maximizing P(data | θ) (03)
     → MAP: add a prior → regularization (04)
     → uncertainty about conclusions: tests and intervals (05)
```
