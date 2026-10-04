# Singular Value Decomposition

## Why It Matters for AI

Eigenvalues only exist for square matrices, but AI data matrices are rectangular — $n$ examples by $d$ features. The **singular value decomposition (SVD)** is the rectangular generalization: every real matrix, any shape, splits into three simple pieces that expose its true rank, its dominant directions, and a recipe for optimal compression. It closes the section's second chain:

```text
Matrix → Linear Transformation → Eigenvalues → SVD → PCA
```

PCA in section 07 is an application of this chapter's content.

## Statement

For any real matrix $A$ with $m$ rows and $n$ columns, there exist:

- $U$: $m \times m$ **orthogonal** matrix (columns orthonormal),
- $\Sigma$: $m \times n$ **diagonal** matrix with entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ on the diagonal,
- $V$: $n \times n$ **orthogonal** matrix,

such that

$$
\boxed{\;A = U \Sigma V^\top\;}
$$

The $\sigma_i$ are the **singular values**; columns $u_i$ of $U$ are **left singular vectors**, columns $v_i$ of $V$ are **right singular vectors**.

Reading: $V^\top$ rotates/reflects the input space, $\Sigma$ scales along the new axes (padding with zero rows/columns when $m \ne n$), $U$ rotates the output space. Any linear map, however mixed, is "rotate → axis-aligned scaling → rotate" — the orthogonal generalization of $A = S\Lambda S^{-1}$ from chapter 06.

Orthogonal matrix: $Q^\top Q = QQ^\top = I$ — columns orthonormal; geometrically a rotation or reflection (lengths and angles preserved).

## How It Is Constructed (From Eigenvalues)

The SVD is built from the eigendecomposition of the symmetric matrix $A^\top A$:

1. $A^\top A$ is $n \times n$, symmetric, positive semi-definite (chapter 06) → eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ orthonormal, eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots \ge 0$.
2. **Right singular vectors** = those eigenvectors: $V = [\mathbf{v}_1 \cdots \mathbf{v}_n]$.
3. **Singular values** = square roots of eigenvalues: $\sigma_i = \sqrt{\lambda_i}$, same descending order.
4. **Left singular vectors** = normalized images: $\mathbf{u}_i = \dfrac{A\mathbf{v}_i}{\sigma_i}$ for $\sigma_i > 0$; extend with any orthonormal completion for $\sigma_i = 0$.

Why step 1 works: $A^\top A \mathbf{v} = \sigma^2 \mathbf{v}$ means $\mathbf{v}$ is an input direction that $A$ maps into something orthogonal-ly ordered; then $A\mathbf{v} = \sigma \mathbf{u}$ with $\|\mathbf{u}\| = 1$ because

$$
\|A\mathbf{v}_i\|^2 = \mathbf{v}_i^\top A^\top A \mathbf{v}_i = \sigma_i^2 \|\mathbf{v}_i\|^2 = \sigma_i^2
$$

so dividing $A\mathbf{v}_i$ by $\sigma_i$ gives a unit vector.

Dimensions to track: $U$ is $m \times m$, $\Sigma$ is $m \times n$, $V^\top$ is $n \times n$; the product $(m \times m)(m \times n)(n \times n)$ gives $m \times n$ ✓

## Worked Example

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

**Step 1 — $A^\top A$.**

$$
A^\top A = \begin{bmatrix}1&0&0\\0&2&0\end{bmatrix}
\begin{bmatrix}1&0\\0&2\\0&0\end{bmatrix}
= \begin{bmatrix}1&0\\0&4\end{bmatrix}
$$

**Step 2 — eigendecomposition** (chapter 06; diagonal → eigenvectors are $\mathbf{e}_1, \mathbf{e}_2$, eigenvalues $1, 4$). Sorted descending: $\lambda_1 = 4$, $\lambda_2 = 1$.

$$
V = \begin{bmatrix}0&1\\1&0\end{bmatrix}
\qquad
(\mathbf{v}_1 = \mathbf{e}_2,\ \mathbf{v}_2 = \mathbf{e}_1)
$$

(the column order must follow descending eigenvalues).

**Step 3 — singular values.** $\sigma_1 = \sqrt{4} = 2$, $\sigma_2 = \sqrt{1} = 1$.

**Step 4 — left singular vectors.**

$$
A\mathbf{v}_1 = \begin{bmatrix}0\\2\\0\end{bmatrix},\;\;
\mathbf{u}_1 = \tfrac{1}{2}\begin{bmatrix}0\\2\\0\end{bmatrix} = \begin{bmatrix}0\\1\\0\end{bmatrix};
\qquad
A\mathbf{v}_2 = \begin{bmatrix}1\\0\\0\end{bmatrix},\;\;
\mathbf{u}_2 = \begin{bmatrix}1\\0\\0\end{bmatrix}
$$

Complete $U$ with a third unit vector orthogonal to both: $\mathbf{u}_3 = (0,0,1)$.

$$
U = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad
\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \quad
V = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

**Verify** $U\Sigma V^\top$:

$$
U\Sigma = \begin{bmatrix}0&1&0\\1&0&0\\0&0&1\end{bmatrix}\begin{bmatrix}2&0\\0&1\\0&0\end{bmatrix} = \begin{bmatrix}0&1\\2&0\\0&0\end{bmatrix}
$$

$$
(U\Sigma)V^\top = \begin{bmatrix}0&1\\2&0\\0&0\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix} = \begin{bmatrix}1&0\\0&2\\0&0\end{bmatrix} = A \;✓
$$

What the decomposition says about this matrix: the strongest input direction is $\mathbf{v}_1 = (0,1)$ (the second feature), mapped to output $(0,1,0)$ with strength $2$; the second direction $\mathbf{v}_2 = (1,0)$ has strength $1$. Rank = number of positive singular values = 2.

### Special case: symmetric positive-definite matrices

If $A$ is symmetric with positive eigenvalues, the SVD coincides with the eigendecomposition: $U = V$ = eigenvector matrix, $\sigma_i = \lambda_i$. For the chapter 06 matrix $\begin{bmatrix}2&1\\1&2\end{bmatrix}$: singular values $3, 1$; $U = V = \tfrac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$.

## Low-Rank Approximation

Keep only the top $k$ singular triplets:

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

a rank-$k$ matrix using $k(m + n + 1)$ numbers instead of $mn$.

**Worked example (truncation).** From the example above, $k = 1$:

$$
A_1 = 2\,\mathbf{u}_1\mathbf{v}_1^\top = 2\begin{bmatrix}0\\1\\0\end{bmatrix}\begin{bmatrix}0&1\end{bmatrix}
= \begin{bmatrix}0&0\\0&2\\0&0\end{bmatrix}
$$

Error: $A - A_1 = \begin{bmatrix}1&0\\0&0\\0&0\end{bmatrix}$, whose only non-zero singular value is $1 = \sigma_2$.

**Eckart–Young theorem** (stated without proof): $A_k$ is the closest rank-$k$ matrix to $A$ in both the Frobenius and spectral norms, and the errors are governed by the discarded singular values:

$$
\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}
$$

$r$ = rank. Dropping a singular value of $0.01$ costs almost nothing; dropping one of $50$ costs a lot. Singular values are the exchange rate for compression.

## SVD, Rank, and $A^\top A$ — Summary

| quantity | how to get it |
|---|---|
| rank of $A$ | count $\sigma_i > 0$ |
| right singular vectors $V$ | eigenvectors of $A^\top A$ |
| left singular vectors $U$ | eigenvectors of $AA^\top$ (or $A\mathbf{v}_i / \sigma_i$) |
| singular values | $\sqrt{\text{eigenvalues of } A^\top A}$ |
| best rank-$k$ approximation | keep top $k$ triplets |

Also: $\operatorname{rank}(A) = \operatorname{rank}(A^\top)$ falls out of this — both share singular values.

## Connection to AI

- **PCA is SVD in disguise** (section 07): apply the SVD to the *centered* data matrix $X$; the right singular vectors $V$ are exactly the principal components. Nobody computes eigenvalues separately in practice — libraries run SVD because it is more stable.
- **Embedding geometry.** The singular values of a word/embedding matrix measure how many "effective dimensions" the embeddings use. A $300$-dimensional embedding space with only $30$ substantial singular values is effectively $30$-dimensional — low intrinsic rank.
- **Compression and distillation.** Truncated SVD compresses weight matrices; the error is provably minimal for the rank kept — the mathematical justification for low-rank adaptation of large models.
- **Latent semantics.** Truncated SVD of a term-document matrix is latent semantic analysis: the top components capture meaning axes that no single feature contains (a concrete instance of "rank = hidden structure").
- **Numerical stability.** $\sigma_{\min} = 0$ signals a singular matrix; tiny $\sigma_{\min}$ signals an *ill-conditioned* one — small input changes amplify, which is why $(X^\top X)^{-1}$ in normal equations can be numerically dangerous and gradient methods (section 05) are used instead.

## Common Mistakes

- Forgetting to order singular values descending — $k$ means "top $k$", so the order is part of the definition.
- Confusing $U$ (left, $m \times m$) with $V$ (right, $n \times n$); $A = U\Sigma V^\top$, not $V\Sigma U^\top$ (wrong shapes when $m \ne n$).
- Assuming $\Sigma$ is square — it is $m \times n$; the diagonal structure is partial when the matrix is rectangular.
- Thinking SVD requires invertibility — it exists for *every* matrix, including singular and rectangular ones. That is its advantage.
- Expecting unique singular vectors: when singular values repeat, the corresponding vectors are determined only up to rotation within their subspace (values themselves are unique).

## Exercises

1. Find the singular values of $\begin{bmatrix}5\\0\end{bmatrix}$ by computing $A^\top A$.
2. Storing $A$ above takes $6$ numbers; storing its rank-1 factors $(\sigma_1, \mathbf{u}_1, \mathbf{v}_1)$ takes $3 + 1 + 2 = 6$ — no saving at this size. For a general $m \times n$ matrix, factors take $k(m+n+1)$ numbers vs. $mn$. Show that $m = n = 1000$, $k = 10$ gives $20{,}010$ vs. $1{,}000{,}000$ and conclude when truncation pays off.
3. Compute the full SVD of $A = \begin{bmatrix}0&0\\3&0\end{bmatrix}$. (Steps: $A^\top A$, eigenvalues, $V$, $\Sigma$, $U$ — note $\sigma_2 = 0$ requires an orthonormal completion.)
4. Show directly that $\|A\mathbf{v}_1\| = \sigma_1$ for the worked example's $\mathbf{v}_1 = (0,1)$.
5. State the rank of a $4 \times 7$ matrix whose singular values are $9, 4, 0, 0, 0, 0, 0$.
6. True or false, with a one-line reason: "If $A$ is invertible, all singular values are non-zero."
