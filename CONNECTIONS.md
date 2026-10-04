# Mathematical Connections

A single idea, once learned, reappears in later sections in a new disguise. This page collects the recurring chains so the connections are visible in one place. Each link is treated in full in the section named below.

## Dot Product

```text
Linear Algebra (01)
      ↓  similarity between vectors
Statistics (04)
      ↓  covariance as an inner product of centered data
Machine Learning (07)
      ↓  logits = wᵀx + b
Transformers (09)
      ↓  attention scores = qᵀk
```

The same operation — multiply components, add — measures angle in 01, scores examples in 07, and decides which tokens attend to which in 09. The matrix form $QK^\top$ in attention computes every pairwise dot product between queries and keys at once.

## Derivative

```text
Calculus (02)
      ↓  gradient ∇J(θ)
Optimization (05)
      ↓  gradient descent update
Neural Networks (08)
      ↓  chain rule: backpropagation
Transformers (09)
      ↓  training W_Q, W_K, W_V by the same rule
```

Backpropagation is not a framework feature. It is the multivariable chain rule applied to a composition of layers, derived in 08 and reused everywhere after.

## Probability

```text
Probability (03)
      ↓  conditional probability P(A | B)
Statistics (04)
      ↓  likelihood of parameters given data
Logistic Regression (07)
      ↓  P(y = 1 | x)
Language Models (10)
      ↓  P(next token | previous tokens)
```

Every classifier outputs a conditional probability; a language model is a conditional probability over a very large alphabet (the vocabulary), conditioned on a long prefix.

## Cross Entropy

```text
Probability (03)          Distribution of the true label
      ↓
Information Theory (06)   H(p, q) = −Σ p(x) log q(x)
      ↓
Classification (07, 08)   Loss for logistic regression and neural networks
      ↓
Language Models (10)      Training objective over token sequences
```

Cross-entropy measures how far a predicted distribution $q$ is from the true distribution $p$. Minimizing it is what "training" means in practice.

## Norms and Distance

```text
Linear Algebra (01)       ‖x‖, distance d(x, y)
      ↓
Statistics (04)           standard deviation is a norm of centered data
      ↓
Optimization (05)         L2 regularization = penalty on ‖θ‖₂
      ↓
Transformers (09)         |qᵀk| ≤ ‖q‖‖k‖; the /√d_k factor tames the scale
```

## Eigenvectors

```text
Linear Algebra (01)       Av = λv
      ↓
Statistics (04)           covariance matrix is symmetric
      ↓
Machine Learning (07)     PCA = eigenvectors of the covariance matrix
      ↓
LLM Mathematics (10)      PCA of embedding matrices exposes linear structure
```

## Chain Rule

```text
Calculus (02)                       d/dx f(g(x)) = f'(g(x))·g'(x)
      ↓
Neural Networks (08)                backpropagation: δ gradients layer by layer
      ↓
Transformers (09)                   gradients flow through attention
      ↓
LLM Training (10)                   gradients of the cross-entropy objective
```

## Softmax

```text
Probability (03)          normalize non-negative weights into a distribution
      ↓
Information Theory (06)  a distribution feeds entropy and cross-entropy
      ↓
Transformers (09)        attention weights = softmax(QKᵀ/√d_k)
      ↓
LLM Mathematics (10)     logits → token probabilities = softmax(logits)
```

Softmax appears twice in a language model: once to turn attention scores into weighted averages, once to turn the final logits into a probability distribution over the vocabulary. Same function, two roles.

## Regularization Across Sections

```text
Statistics (04)    MAP estimation with a Gaussian prior
      ↓
Optimization (05)  adding ‖θ‖² to the objective
      ↓
Machine Learning (07)  ridge regression = closed form of that penalty
      ↓
Neural Networks (08)   weight decay during training
```

## The Whole Path at Once

```text
tokens → embeddings (09, 10) → attention with dot products (09)
       → logits (07, 10) → softmax (03, 07) → distribution over next token (03)
       → cross-entropy loss (06) → gradient (02, 08) → update (05, 08)
       → repeat
```

If you can explain every term in that line, you understand the mathematics of a language model.
