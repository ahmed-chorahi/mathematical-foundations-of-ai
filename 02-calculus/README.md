# 02 — Calculus

Calculus is the mathematics of change, and training a model is entirely about change: how does the loss move when a parameter moves, and in which direction does it move least? This section builds that ability step by step and ends with the chain rule — the single most important tool for understanding how neural networks learn.

## The Chain

```text
Function → Derivative → Partial Derivatives → Gradient → Chain Rule → Backpropagation
```

## Contents

| File | Topic | AI application |
|---|---|---|
| [01-limits-and-continuity.md](01-limits-and-continuity.md) | limits, continuity, what differentiability needs | smooth vs non-smooth activations |
| [02-derivatives.md](02-derivatives.md) | derivative, rules, common derivatives, sigmoid derivative | rates of change of loss and activations |
| [03-partial-derivatives-and-gradient.md](03-partial-derivatives-and-gradient.md) | partials, gradient, directional derivative | steepest descent direction |
| [04-jacobian-and-hessian.md](04-jacobian-and-hessian.md) | Jacobian and Hessian matrices | per-parameter sensitivities, curvature |
| [05-chain-rule.md](05-chain-rule.md) | univariate and multivariate chain rule, computational graphs | backpropagation |
| [06-taylor-approximation.md](06-taylor-approximation.md) | first- and second-order expansion | why gradient descent takes a step |
| [07-optimization-with-calculus.md](07-optimization-with-calculus.md) | critical points, second-derivative test | locating minima of the loss |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Section 00 (functions, exponents, logarithms) and basic familiarity with vectors and matrices from section 01. Multivariable chapters use $\nabla$, gradients, and matrix notation but no further linear-algebra theory.

## Conventions

- $f'(x)$ and $\dfrac{df}{dx}$ are the same derivative; $\dfrac{\partial f}{\partial x}$ is a partial derivative.
- Gradients are column vectors: $\nabla f = \left(\dfrac{\partial f}{\partial x_1}, \ldots, \dfrac{\partial f}{\partial x_n}\right)^\top$.
- Unless stated, functions are differentiable where evaluated.
