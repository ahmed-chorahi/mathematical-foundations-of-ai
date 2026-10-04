# Mathematical Foundations of AI

A structured curriculum in the mathematics that underlies artificial intelligence: linear algebra, calculus, probability, statistics, optimization, information theory, and their application to machine learning, neural networks, transformers, and large language models.

This repository is a mathematics resource. It is not an AI framework, not a programming tutorial, and not a collection of tool guides. Every chapter exists to build one capability: reading the mathematics of modern AI and understanding why it works.

## Who This Is For

The intended reader is an engineer or student who:

- can handle basic algebra and wants the mathematics behind machine learning,
- has seen formulas like $\nabla J(\theta)$ or $\operatorname{softmax}(QK^\top/\sqrt{d_k})$ and wants to know where they come from,
- prefers correct, sequenced explanations to short summaries.

No prior linear algebra, calculus, or probability is assumed. Programming knowledge helps for the embedded Python examples but is not required to follow the mathematics.

## Learning Order

Work through the sections in order. Each depends on the ones before it.

```text
00  Prerequisites
 ↓
01  Linear Algebra
 ↓
02  Calculus
 ↓
03  Probability
 ↓
04  Statistics
 ↓
05  Optimization
 ↓
06  Information Theory
 ↓
07  Machine Learning Mathematics   ← combines 01–06
 ↓
08  Neural Network Mathematics
 ↓
09  Transformer Mathematics
 ↓
10  LLM Mathematics
```

The full dependency diagram and per-section goals are in [ROADMAP.md](ROADMAP.md). The map of how individual concepts reappear across sections is in [CONNECTIONS.md](CONNECTIONS.md).

## What Is Covered

| Section | Mathematics | Where it appears in AI |
|---|---|---|
| 00 Prerequisites | algebra, functions, exponents, logarithms, summation | the baseline notation used everywhere |
| 01 Linear Algebra | vectors, matrices, dot products, norms, rank, basis, projections, least squares, eigenvalues, SVD | feature vectors, embeddings, weights, linear layers, similarity, attention, PCA |
| 02 Calculus | limits, derivatives, partial derivatives, gradients, Jacobian, Hessian, chain rule, Taylor approximation | loss functions, gradient descent, backpropagation, parameter updates |
| 03 Probability | conditional probability, Bayes' theorem, random variables, distributions, expectation, covariance | uncertainty, classification, generative models, token probabilities |
| 04 Statistics | estimators, bias, likelihood, maximum likelihood, MAP, hypothesis tests, confidence intervals | fitting models from data, regularization, model evaluation |
| 05 Optimization | objective functions, gradient descent, convexity, regularization, Lagrange multipliers | training: every model is fitted by minimizing something |
| 06 Information Theory | entropy, cross-entropy, KL divergence, mutual information | classification losses, language-model training, model uncertainty |
| 07 ML Mathematics | linear regression, logistic regression, PCA, derived from the earlier sections | classical models, built by hand from the pieces above |
| 08 Neural Networks | neuron, layers, activations, loss, chain rule, backpropagation | how a network is trained |
| 09 Transformers | embeddings, Q/K/V, self-attention, softmax, multi-head structure | attention, the core mechanism of modern architectures |
| 10 LLMs | logits, softmax, next-token probability, cross-entropy, perplexity, temperature | language-model training and generation |

## How Each Chapter Is Written

Chapters follow a consistent pattern where it helps:

1. **Why it matters for AI** — the problem the mathematics solves.
2. **Definition and intuition** — what the object is, in plain language, with notation.
3. **Derivation** — where the important formulas come from, step by step.
4. **Worked example** — small numbers you can follow by hand.
5. **Connection to AI** — the specific place the concept is used.
6. **Common mistakes** — where understanding usually breaks.
7. **Exercises** — a few conceptual, computational, and implementation problems; solutions with full reasoning are at the end of each section in `SOLUTIONS.md`.

## Rules This Repository Follows

- **Mathematics first.** Every concept is introduced because something in AI needs it, not because it exists.
- **Small examples.** Vectors of length 2, matrices of size $2 \times 2$, distributions over 3 outcomes. Every computation is followable by hand.
- **Light dependencies.** Markdown, LaTeX, and small pure-Python snippets. A chapter is complete if you only read it. Nothing needs to be installed.
- **Correct and honest.** Derivations are checked; approximations are labeled as approximations; no result is asserted without justification.

## How to Use This Repository

- Read in order (see [ROADMAP.md](ROADMAP.md)).
- Do the exercises before reading the solutions.
- Run the Python snippets when you want to check a computation; they are short and use only the standard library.
- Use [CONNECTIONS.md](CONNECTIONS.md) when you want to see how a concept from one section reappears later.

```bash
# no installation required; snippets run with plain Python 3
python3 -c "print(sum(x*y for x, y in zip([1,2],[3,4])))"   # 11
```
