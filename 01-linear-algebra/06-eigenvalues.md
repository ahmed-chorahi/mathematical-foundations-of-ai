# Eigenvalues and Eigenvectors

## Why It Matters for AI

Some directions matter more than others. In a covariance matrix, the top eigenvector is the direction of greatest variance — PCA is built on this. In the Hessian of a loss, eigenvalues describe curvature in each direction and decide whether a critical point is a minimum. Eigenvalues also reveal whether a transformation stretches, shrinks, or flips a space, which is how stability is assessed. This chapter builds the matrix chain's second link:

```text
Matrix → Linear Transformation → Eigenvalues / Eigenvectors → SVD → PCA
```

## Definition

For a square matrix $A$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ when

$$
\boxed{\;A\mathbf{v} = \lambda\mathbf{v}\;}
$$

The transformation $A$ acts on $\mathbf{v}$ exactly like scaling by the number $\mathbf{v}$ — direction unchanged (reversed only if $\lambda < 0$), length multiplied by $|\lambda|$.

Out of all possible inputs, eigenvectors are the ones $A$ treats as simple numbers. Everything else mixes directions together; eigenvectors are the axes of the transformation.

$\mathbf{v} = \mathbf{0}$ is excluded because $A\mathbf{0} = \lambda\mathbf{0}$ holds trivially for every $\lambda$ — it would make every number an eigenvalue.

## Finding Eigenvalues

Rearrange the definition:

$$
A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}
\quad\Rightarrow\quad
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

Subtracting $\lambda \mathbf{v}$ requires the identity matrix to keep the shapes valid: $A\mathbf{v} - \lambda I\mathbf{v}$.

Now: does $(A - \lambda I)\mathbf{v} = \mathbf{0}$ have a solution with $\mathbf{v} \ne \mathbf{0}$? By chapter 04, a homogeneous system has a non-zero solution exactly when the matrix is **singular**:

$$
\boxed{\;\det(A - \lambda I) = 0\;}
$$

This is the **characteristic equation**. For an $n \times n$ matrix it is a degree-$n$ polynomial in $\lambda$, so there are exactly $n$ eigenvalues counted with multiplicity (some may be complex — a separate fact from being *distinct*).

## Worked Example

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

**Step 1 — characteristic polynomial.**

$$
\det(A - \lambda I) = \det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}
= (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3)
$$

Roots: $\lambda_1 = 1$, $\lambda_2 = 3$.

**Step 2 — eigenvector for $\lambda = 1$.**

$$
(A - I)\mathbf{v} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\mathbf{v} = \mathbf{0}
\quad\Rightarrow\quad v_1 + v_2 = 0
\quad\Rightarrow\quad
\mathbf{v}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

**Step 3 — eigenvector for $\lambda = 3$.**

$$
(A - 3I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\mathbf{v} = \mathbf{0}
\quad\Rightarrow\quad v_1 = v_2
\quad\Rightarrow\quad
\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

**Verify.**

$$
A\mathbf{v}_1 = \begin{bmatrix}2&1\\1&2\end{bmatrix}\begin{bmatrix}1\\-1\end{bmatrix}
= \begin{bmatrix}1\\-1\end{bmatrix} = 1\cdot\mathbf{v}_1 \;✓
\qquad
A\mathbf{v}_2 = \begin{bmatrix}3\\3\end{bmatrix} = 3\cdot\mathbf{v}_2 \;✓
$$

Read geometrically: along $(1,-1)$ this matrix does nothing (scale 1); along $(1,1)$ it stretches by 3.

## Diagonalization

When $A$ has $n$ independent eigenvectors, stack them as columns of $S$ and eigenvalues as a diagonal $\Lambda$:

$$
AS = S\Lambda
\quad\Longleftrightarrow\quad
A = S\Lambda S^{-1}
$$

($j$-th column: $A\mathbf{s}_j = \lambda_j \mathbf{s}_j$ — exactly $AS = S\Lambda$ column by column.)

Powers become trivial:

$$
A^k = S\Lambda^k S^{-1}
$$

because $\Lambda^k$ just raises each diagonal entry to the power $k$; the $S$ factors cancel in pairs.

**Check with the example.** $S = \begin{bmatrix}1&1\\1&-1\end{bmatrix}$, $\Lambda = \operatorname{diag}(3, 1)$, $S^{-1} = \tfrac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$.

$A^2 = S\Lambda^2 S^{-1} = \begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}9&0\\0&1\end{bmatrix}\tfrac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix} = \begin{bmatrix}5&4\\4&5\end{bmatrix}$

Direct check: $\begin{bmatrix}2&1\\1&2\end{bmatrix}^2 = \begin{bmatrix}5&4\\4&5\end{bmatrix}$ ✓

Why it matters beyond this example: repeated application of a linear map (time steps of a dynamical system, iterated weight updates) becomes multiplication of eigenvalues rather than of matrices.

## Symmetric Matrices

$A$ is **symmetric** when $A = A^\top$. These are the matrices of AI:

- $X^\top X$ (normal equations, chapter 05),
- covariance matrices (section 04),
- Gram/similarity matrices $XX^\top$,
- the Hessian of a scalar function (section 02) is symmetric whenever second derivatives commute.

Symmetry buys two theorems (stated without proof — the proofs are standard linear algebra):

1. **All eigenvalues are real.** No complex numbers to worry about.
2. **Eigenvectors for different eigenvalues are orthogonal.** Distinct eigenvalue directions meet at exactly $90°$.

Together: every symmetric matrix has an orthonormal eigenbasis — a perfect coordinate system aligned with the transformation. This is the **spectral theorem**, and it is why PCA (section 07) works on covariance matrices with no complications.

Example check: our $A$ is symmetric; eigenvectors $(1,1)$ and $(1,-1)$ satisfy $(1)(1) + (1)(-1) = 0$ ✓.

## Positive-Definite Matrices

For symmetric $A$, the **quadratic form** $Q(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ is a single number per vector. (Expand for $2 \times 2$: if $A = \begin{bmatrix}a&b\\b&c\end{bmatrix}$ then $\mathbf{x}^\top A\mathbf{x} = ax_1^2 + 2bx_1x_2 + cx_2^2$.)

| property | condition | meaning of $\mathbf{x}^\top A\mathbf{x}$ |
|---|---|---|
| **positive definite** | $> 0$ for all $\mathbf{x} \ne \mathbf{0}$ | always positive — like an energy |
| positive semi-definite | $\ge 0$ for all $\mathbf{x}$ | non-negative; zero allowed |
| indefinite | sometimes positive, sometimes negative | sign varies with direction |

**Eigenvalue test** (symmetric $A$): $A$ is positive definite $\iff$ every eigenvalue is positive.

*Why:* write $\mathbf{x} = \sum_i c_i \mathbf{v}_i$ in the orthonormal eigenbasis; then $\mathbf{x}^\top A\mathbf{x} = \sum_i \lambda_i c_i^2$, a weighted sum of eigenvalues with non-negative weights $c_i^2$ — positive for all $\mathbf{x} \ne \mathbf{0}$ exactly when all $\lambda_i > 0$.

**Worked example — positive definite.** $A = \begin{bmatrix}2&0\\0&3\end{bmatrix}$: $\mathbf{x}^\top A\mathbf{x} = 2x_1^2 + 3x_2^2 > 0$ for $\mathbf{x} \ne \mathbf{0}$. Eigenvalues $2, 3 > 0$ ✓

**Worked example — not.** $B = \begin{bmatrix}1&2\\2&1\end{bmatrix}$ (eigenvalues $3$ and $-1$): with $\mathbf{x} = (1,-1)$, first compute $B\mathbf{x} = \begin{bmatrix}1-2\\2-1\end{bmatrix} = \begin{bmatrix}-1\\1\end{bmatrix}$, then

$$
\mathbf{x}^\top(B\mathbf{x}) = (1)(-1) + (-1)(1) = -2 < 0
$$

A negative direction exists → indefinite, not positive definite.

Also: $A$ positive definite $\Rightarrow$ all pivots positive in elimination $\Rightarrow$ $A$ invertible and $A = LL^\top$ (Cholesky) — the practical reasons optimizers and covariance code test for it.

Why AI cares about positive definiteness:

- A symmetric matrix from a covariance is positive **semi**definite: $\mathbf{x}^\top \Sigma \mathbf{x} = \operatorname{Var}(\mathbf{x}^\top \text{data}) \ge 0$ always — variances cannot be negative.
- A loss near a minimum curves upward in every direction — the Hessian is positive definite there (section 05). Positive definite = a true local minimum candidate.
- Newton's method divides by curvature (the Hessian); positive definiteness guarantees the division points downhill.

## Connection to AI

- **PCA** (section 07) = eigenvectors of the covariance matrix, ordered by eigenvalue. The whole algorithm is "symmetric matrix → spectral theorem → orthonormal directions ranked by variance".
- **Covariance structure.** Eigenvalues of $\Sigma$ = variance along each principal direction; large gaps = the data effectively has fewer dimensions (low intrinsic rank).
- **Curvature and optimization.** The Hessian's eigenvalues give the sharpest and flattest curvature directions at a critical point; positive definite ⇔ local minimum (second-derivative test in matrix form, section 05).
- **Stability.** Iterating a linear recurrence $\mathbf{x}_{t+1} = A\mathbf{x}_t$: the response grows or decays like $\lambda^t$. $|\lambda| < 1$ contracts, $|\lambda| > 1$ explodes — the criterion for exploding/vanishing signals in deep products of matrices (section 08 references this intuition).
- **Low-rank structure.** Only the largest eigenvalues matter in practice — truncation is compression (leads directly into SVD next).

## Common Mistakes

- Forgetting $\mathbf{v} \ne \mathbf{0}$ — without it the definition is vacuous.
- Dividing $\det(A - \lambda I)$ incorrectly: the $\lambda I$ subtraction is on the *diagonal only*.
- Assuming every matrix has a full set of real eigenvectors — false in general (rotation matrices have none in $\mathbb{R}^2$); symmetry is what guarantees the nice behavior, and it must be assumed.
- Treating eigenvalues as sorted or unique: order them yourself when you need "the largest".
- Confusing $\mathbf{x}^\top A\mathbf{x}$ (a scalar quadratic form) with $A\mathbf{x}$ (a vector).

## Exercises

1. Verify that $\mathbf{v} = (1, -1)$ is an eigenvector of $A = \begin{bmatrix}2&1\\1&2\end{bmatrix}$ and find its eigenvalue.
2. Find the eigenvalues of $\begin{bmatrix}3&0\\0&5\end{bmatrix}$ without solving a polynomial — explain why diagonal matrices make this immediate.
3. Find all eigenvalues and eigenvectors of $\begin{bmatrix}4&1\\0&3\end{bmatrix}$. Is it symmetric?
4. Determine whether $C = \begin{bmatrix}2&1\\1&2\end{bmatrix}$ is positive definite using the eigenvalue test (reuse exercise 1's eigenvalues).
5. Show that if $A\mathbf{v} = \lambda\mathbf{v}$ then $A^2\mathbf{v} = \lambda^2\mathbf{v}$. State the general pattern for $A^k$.
6. A symmetric matrix has eigenvalues $-2$ and $4$. Is it positive definite? What can you say about $\mathbf{x}^\top A\mathbf{x}$ for some $\mathbf{x}$?
7. Compute $\mathbf{x}^\top B \mathbf{x}$ for $B = \begin{bmatrix}1&0\\0&-1\end{bmatrix}$ and $\mathbf{x} = (0,1)$; conclude whether $B$ is positive definite.
