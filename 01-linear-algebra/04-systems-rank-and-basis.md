# Systems, Rank, and Basis

## Why It Matters for AI

Two questions run through all of AI: *does a solution exist?* and *is it unique?* Can this model fit the data? Are its parameters determined by the data, or are some directions unconstrained? Is a dataset redundant? These are questions about systems of linear equations, and the concepts of span, independence, basis, and rank are the precise language for answering them.

## Systems of Linear Equations

A system collects equations that share unknowns:

$$
\begin{cases}
x_1 + 2x_2 = 5 \\
3x_1 + 4x_2 = 11
\end{cases}
$$

Every equation is **linear**: unknowns appear only to the first power, multiplied by constants and added.

In matrix form the system is $A\mathbf{x} = \mathbf{b}$:

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 5 \\ 11 \end{bmatrix}
$$

This single line encodes the whole system because matrix–vector multiplication builds each equation as a dot product of a row with $\mathbf{x}$ (chapter 03).

### Solving: elimination

Multiply a row by a constant, add a multiple of one row to another (these operations do not change the solution set), until the matrix becomes upper triangular:

$$
\begin{bmatrix} 1 & 2 & 5 \\ 3 & 4 & 11 \end{bmatrix}
\;\xrightarrow{\;R_2 \mathrel{-=} 3R_1\;}\;
\begin{bmatrix} 1 & 2 & 5 \\ 0 & -2 & -4 \end{bmatrix}
\;\xrightarrow{\;R_2 \mathrel{/=} (-2)\;}\;
\begin{bmatrix} 1 & 2 & 5 \\ 0 & 1 & 2 \end{bmatrix}
$$

Back-substitution: $x_2 = 2$, then $x_1 = 5 - 2(2) = 1$. Check: $3(1) + 4(2) = 11$ ✓

Elimination is not just a high-school trick — it *counts pivots*, and the pivot count is the rank below.

### Three possibilities

For $A\mathbf{x} = \mathbf{b}$:

| case | meaning | in elimination |
|---|---|---|
| exactly one solution | $\mathbf{b}$ reachable, constraints sufficient | a pivot in every column |
| no solution | $\mathbf{b}$ not reachable | inconsistent row like $[0 \mid c]$, $c \ne 0$ |
| infinitely many solutions | reachable but under-constrained | a column without a pivot |

**Worked example (no solution).** $x_1 + x_2 = 1$ and $x_1 + x_2 = 2$ — the same left side, different right sides; subtraction gives $0 = 1$. The lines are parallel and distinct.

**Worked example (infinitely many).** $x_1 + x_2 = 1$: every $(x_1, 1-x_1)$ works — a whole line of solutions.

## Linear Combination and Span

The **span** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is the set of every linear combination:

$$
\operatorname{span}(\mathbf{v}_1,\ldots,\mathbf{v}_k)
= \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}
$$

- $\operatorname{span}(\mathbf{v})$ for one non-zero vector = the line through the origin in direction $\mathbf{v}$.
- $\operatorname{span}(\mathbf{v}_1, \mathbf{v}_2)$ for non-parallel vectors = a plane through the origin.

Key fact: $\operatorname{span}$ of $k$ vectors in $\mathbb{R}^n$ has dimension at most $k$ — you cannot manufacture more than $k$ degrees of freedom from $k$ vectors.

**Span and existence.** $A\mathbf{x} = \mathbf{b}$ has a solution **exactly when $\mathbf{b} \in \operatorname{span}(\text{columns of } A)$** — because $A\mathbf{x}$ *is* a linear combination of the columns (chapter 03). Column $j$ of $A$ scaled by $x_j$, summed, must land on $\mathbf{b}$.

## Linear Independence

Vectors are **linearly dependent** if one of them is a combination of the others — a non-trivial combination gives the zero vector:

$$
c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0}
\quad\text{with not all } c_i = 0
$$

Otherwise they are **linearly independent**: the only way to combine them to zero is with all coefficients zero.

**Worked example.** $\mathbf{v}_1 = (1, 0)$, $\mathbf{v}_2 = (2, 0)$: $2\mathbf{v}_1 - \mathbf{v}_2 = \mathbf{0}$ — dependent (both lie on the x-axis; the second adds no new direction).

$\mathbf{e}_1 = (1,0)$, $\mathbf{e}_2 = (0,1)$: $c_1\mathbf{e}_1 + c_2\mathbf{e}_2 = (c_1, c_2) = \mathbf{0}$ forces $c_1 = c_2 = 0$ — independent.

**Independence and uniqueness.** For $A\mathbf{x} = \mathbf{b}$: unique solution exactly when the columns are independent. Dependence means some column is a combination of the rest — you can add that combination to $\mathbf{x}$ and compensate elsewhere, producing another valid solution (infinitely many).

## Basis and Dimension

A **basis** of a space (or subspace) is a set of vectors that is:

1. **linearly independent** — no redundancy, and
2. **spanning** — everything is reachable.

So a basis is a minimal complete set of directions: enough to reach every point, with no spare copies.

**Dimension** = number of vectors in any basis.

Examples:

| space | basis | dimension |
|---|---|---|
| $\mathbb{R}^2$ | $\{(1,0), (0,1)\}$ | 2 |
| $\mathbb{R}^n$ | standard basis $\mathbf{e}_1,\ldots,\mathbf{e}_n$ | $n$ |
| line through origin in $\mathbb{R}^2$ | any single non-zero vector on it | 1 |
| plane through origin in $\mathbb{R}^3$ | any two non-parallel vectors in it | 2 |

Bases are not unique (any non-zero vector is a basis of its own line) — **dimension is**. Changing basis without changing the space is the operation behind representations: same data, different coordinates (PCA, section 07, picks a basis that exposes variance).

## Vector Spaces and Subspaces (The Structure)

A **vector space** is a set closed under vector addition and scalar multiplication, with the properties listed in chapter 01 (commutativity of $+$, existence of $\mathbf{0}$, etc.). $\mathbb{R}^n$ is the prototype; the main examples for this course are:

- the span of some vectors (always contains $\mathbf{0}$),
- solution sets of homogeneous systems $A\mathbf{x} = \mathbf{0}$,
- the row space and column space of a matrix.

A **subspace** of $\mathbb{R}^n$ is a subset that is itself a vector space — equivalently: contains $\mathbf{0}$, closed under addition and scaling.

**Test.** The line $y = 2x$ through the origin is a subspace of $\mathbb{R}^2$. The line $y = 2x + 1$ is **not**: it misses $\mathbf{0}$ (and $2(1,3) + (1,3)$ leaves the line). This is why homogeneous systems give subspaces and inhomogeneous ones give translated copies.

Why subspaces matter: "the directions a model can reach" is a subspace question. Layer outputs after a low-rank projection fill only a subspace of the representation space; exploring what is reachable and what is lost is exactly a subspace analysis.

## Rank

The **rank** of a matrix is

$$
\operatorname{rank}(A) = \dim(\text{column space of } A) = \text{number of pivots}
$$

It equals the row dimension too (row rank = column rank — a theorem; the practical procedure is elimination, which counts pivots).

- $A$ is $m \times n$: $\operatorname{rank}(A) \le \min(m, n)$.
- $\operatorname{rank}(A) = n$ (full column rank) → columns independent → $A\mathbf{x} = \mathbf{b}$ has at most one solution.
- $\operatorname{rank}(A) = m$ (full row rank) → a solution exists for every $\mathbf{b}$.
- $A$ **invertible** $\iff$ $\operatorname{rank}(A) = n$ (for square $A$) $\iff$ $\det A \ne 0$ $\iff$ columns independent. All the same fact in four dialects.

**Worked example.** $A = \begin{bmatrix}1 & 2\\2 & 4\end{bmatrix}$ (from the previous chapter): row 2 is $2\times$ row 1, so elimination leaves one pivot. $\operatorname{rank}(A) = 1 < 2$: singular, no inverse, the image is a line rather than the plane.

**Rank as compression.** An $m \times n$ matrix of rank $r < \min(m,n)$ stores only $r$ independent directions — it can be written with $r$ instead of $mn$ numbers. Low-rank structure is why SVD (chapter 07) compresses and why embeddings can be reduced.

## Connection to AI

- **Existence and uniqueness of solutions.** Training solves equations (normal equations in section 07): rank tells you whether the parameters are determined by the data. Rank-deficient design matrices produce non-unique fits — the origin of regularization (section 05).
- **Identifiability.** Two parameters that always move together are linearly dependent in the model — one is redundant. Depth makes networks hugely over-parameterized: far more parameters than data, so solutions exist but are not unique; gradient descent picks one (section 08).
- **Data redundancy.** Correlated features are dependent columns of $X$; PCA (section 07) finds a smaller basis capturing the same span. Rank = intrinsic dimension of a dataset.
- **Reachable representations.** A layer with weight matrix of rank $r$ maps all inputs into an $r$-dimensional subspace — information in the discarded directions is gone.
- **The homogeneous system $A\mathbf{x} = \mathbf{0}$** (the null space) reappears as "which changes leave the output unchanged" — flat directions of a loss, non-unique parameters, and (in transformers) directions in the residual stream that attention may ignore.

## Common Mistakes

- Confusing span with the whole space: $n$ vectors in $\mathbb{R}^n$ span $\mathbb{R}^n$ **only if** they are independent.
- Treating dependence as a defect. Dependence means *redundancy*, not error — over-parameterized networks are deliberately full of it.
- Thinking the basis is unique. It is not; dimension is.
- Forgetting that span must contain $\mathbf{0}$: vectors $\mathbf{v}$ and $2\mathbf{v}$ span a line, not a plane.
- Mixing row rank and column rank numerically — they are equal, but you compute them by reducing the matrix (they are not the transpose's independent count *by accident*; the theorem says they match).

## Exercises

1. Solve by elimination: $2x_1 + x_2 = 7$, $x_1 - x_2 = 2$.
2. Solve $A\mathbf{x} = \mathbf{b}$ for $A = \begin{bmatrix}1&2\\2&4\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}3\\6\end{bmatrix}$ — show there are infinitely many solutions, and find two.
3. Determine whether $(1, 1, 0)$ and $(0, 2, 1)$ are linearly independent.
4. State the span of $(1,0)$ and $(1,0)$; then the span of $(1,0)$ and $(0,1)$.
5. Is $\{(1,0,0), (0,1,0)\}$ a basis of $\mathbb{R}^3$? Explain both conditions.
6. Find the rank of $\begin{bmatrix}1&3&0\\2&6&1\\0&0&2\end{bmatrix}$ by elimination.
7. Let $H = \{\mathbf{x} \in \mathbb{R}^3 : x_1 + x_2 + x_3 = 0\}$. Show $H$ is a subspace (check $\mathbf{0} \in H$, closure under addition, closure under scaling), then show $(1,-1,0)$ and $(0,1,-1)$ are independent solutions spanning $H$, so $\dim H = 2$.
