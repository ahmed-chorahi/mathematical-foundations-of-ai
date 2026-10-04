# 05 — Optimization

Everything a model learns, it learns by optimizing: a scalar objective is defined, and parameters are adjusted to reduce it. This section develops that operation from the ground up — what is being minimized, why the gradient gives the direction, how step size is chosen, when the answer is guaranteed optimal (convexity), how data can be subsampled (stochastic gradients), how priors become penalties (regularization), and what happens when constraints are imposed (Lagrange multipliers).

## Contents

| File | Topic | AI application |
|---|---|---|
| [01-objective-and-loss.md](01-objective-and-loss.md) | objective functions, loss vs objective, min/max | what training minimizes |
| [02-gradient-descent.md](02-gradient-descent.md) | the update rule derived, learning rate, worked iterations | the core training algorithm |
| [03-convexity.md](03-convexity.md) | convex functions, local vs global minima | when training is guaranteed |
| [04-stochastic-gradient-descent.md](04-stochastic-gradient-descent.md) | minibatch gradients as estimates, noise | training on large datasets |
| [05-regularization.md](05-regularization.md) | L2/L1 penalties, weight decay, overfitting | controlling variance |
| [06-constrained-optimization.md](06-constrained-optimization.md) | constraints, Lagrange multipliers | SVM margins, constrained objectives |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Section 02 (gradient, Taylor approximation — chapter 06 there contains the core derivation referenced here), section 04 (likelihoods as the source of losses, MAP as the source of penalties).

## The Central Formula

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta_{\text{old}})
$$

Derived and exercised in [02-gradient-descent.md](02-gradient-descent.md); every other chapter modifies, qualifies, or surrounds it.
