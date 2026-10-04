# 07 — Machine Learning Mathematics

This section is where every previous section combines. Three foundational models, each derived end-to-end from tools already built: **linear regression** (linear algebra's projection + calculus' gradients + statistics' Gaussian likelihood), **logistic regression** (probability's conditional model + information theory's cross-entropy + optimization's convexity), and **PCA** (statistics' covariance + section 01's eigendecomposition + section 05's constrained optimization).

## Contents

| File | Topic | Tools reused |
|---|---|---|
| [01-linear-regression.md](01-linear-regression.md) | model, normal equations, gradient descent, worked fit | projection, convexity, MSE as MLE |
| [02-logistic-regression.md](02-logistic-regression.md) | sigmoid, boundary, cross-entropy loss, gradient | conditional probability, BCE, convexity |
| [03-pca.md](03-pca.md) | covariance, eigendecomposition, projection, variance explained | eigenvectors, Lagrange multipliers |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Everything from sections 00–06 — this section is the integration point the roadmap promised.
