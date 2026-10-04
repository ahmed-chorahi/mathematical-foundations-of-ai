# 01 — Linear Algebra

Linear algebra is the language of AI. Data points are vectors, parameters are matrices, layers are matrix multiplications, similarities are dot products, and attention is a matrix product of learned projections. This section builds the subject as one connected structure rather than a list of definitions.

## The Two Chains

Everything in this section connects along two paths:

```text
Vector → Dot Product → Projection → Orthogonality → Least Squares
   ↓
Matrix → Linear Transformation → Eigenvalues → SVD → PCA
```

The first chain builds regression (section 07); the second builds dimensionality reduction and underpins how networks transform representations.

## Contents

| File | Topic | AI application |
|---|---|---|
| [01-vectors.md](01-vectors.md) | scalars, vectors, addition, scalar multiplication | feature vectors, embeddings |
| [02-dot-product-and-norms.md](02-dot-product-and-norms.md) | dot product, norms, distance, angle, cosine similarity | similarity, attention scores |
| [03-matrices.md](03-matrices.md) | matrices, multiplication, transpose, identity, linear transformations | weights, linear layers |
| [04-systems-rank-and-basis.md](04-systems-rank-and-basis.md) | systems, independence, span, basis, dimension, rank | identifiability, redundancy in data |
| [05-orthogonality-and-least-squares.md](05-orthogonality-and-least-squares.md) | orthogonal projection, least squares, normal equations | linear regression, compression |
| [06-eigenvalues.md](06-eigenvalues.md) | eigenvalues, eigenvectors, symmetric and positive-definite matrices | covariance structure, stability, PCA |
| [07-svd.md](07-svd.md) | singular value decomposition | PCA, low-rank approximation, embeddings analysis |

Exercises are at the end of each chapter; full reasoning is in [SOLUTIONS.md](SOLUTIONS.md).

## Conventions

- Vectors are **column vectors** by default; $\mathbf{x} \in \mathbb{R}^n$ has $n$ real components.
- Vectors are written in bold lowercase ($\mathbf{x}, \mathbf{w}$) or with an arrow ($\vec{x}$); matrices in uppercase ($W, A$).
- $A^\top$ is the transpose; $\|\mathbf{x}\|$ is the Euclidean norm unless stated otherwise.
- Matrix dimensions are always tracked: an $(m \times n)$ times $(n \times p)$ product is legal and yields $(m \times p)$.
