# 03 — Probability

Probability is the mathematics of incomplete information — the natural language of a model that must predict outcomes it cannot see. Classification outputs probabilities, language models are conditional distributions over tokens, and training objectives are expectations. This section builds probability from axioms to distributions, with small numerical examples throughout.

## The Progression

```text
Events and axioms → Conditional probability → Bayes' theorem
      ↓
Random variables → Distributions (PMF/PDF/CDF)
      ↓
Expectation, variance, covariance → Joint, marginal, conditional distributions
```

## Contents

| File | Topic | AI application |
|---|---|---|
| [01-basics.md](01-basics.md) | sample spaces, events, axioms, addition rules | counting outcomes, base rates |
| [02-conditional-and-bayes.md](02-conditional-and-bayes.md) | conditioning, independence, Bayes' theorem | classifiers, priors, uncertainty updating |
| [03-random-variables.md](03-random-variables.md) | RVs, PMF, PDF, CDF, common distributions | labels, noise, softmax outputs |
| [04-expectation-variance.md](04-expectation-variance.md) | expectation, variance, covariance, correlation | expected loss, uncertainty, feature relationships |
| [05-joint-marginal-conditional.md](05-joint-marginal-conditional.md) | joint/marginal/conditional distributions, chain rule | language-model factorization, dependence |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Conventions

- Outcomes live in a sample space $\Omega$; events are subsets of $\Omega$.
- $P(A)$, $P(A \mid B)$, $P(A \cap B)$ are probabilities of events; lowercase $p(x)$ for probabilities of random-variable values.
- $\log$ is base 2 in this section only when information units (bits) are mentioned; otherwise natural log (section 06 makes the base question explicit).
- Random variables are capital letters ($X, Y$); specific values are lowercase ($x$).
