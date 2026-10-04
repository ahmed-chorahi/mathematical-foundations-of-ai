# Dot Product and Norms

## Why It Matters for AI

One operation does most of the work in AI: multiply components, add the results. The **dot product** scores how similar two vectors are; the **norm** measures how long a vector is. Attention scores are dot products, cosine similarity is a normalized dot product, and squared error is built from norms. If you master this chapter, most of modern AI's mechanics will look familiar.

## Definition

For two vectors of the same length $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$:

$$
\mathbf{x} \cdot \mathbf{y} \;=\; \sum_{i=1}^{n} x_i y_i \;=\; x_1y_1 + x_2y_2 + \cdots + x_ny_n
$$

Also written $\mathbf{x}^\top \mathbf{y}$ (row times column — chapter 03) or $\langle \mathbf{x}, \mathbf{y} \rangle$.

The result is a **scalar** — two vectors in, one number out.

**Worked example.**

$$
\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}
\quad\Rightarrow\quad
\mathbf{x}\cdot\mathbf{y} = 1\cdot4 + 2\cdot5 + 3\cdot6 = 4 + 10 + 18 = 32
$$

```python
def dot(x, y):
    return sum(a * b for a, b in zip(x, y))

print(dot([1, 2, 3], [4, 5, 6]))   # 32
```

## Properties

| property | identity | reading |
|---|---|---|
| commutative | $\mathbf{x}\cdot\mathbf{y} = \mathbf{y}\cdot\mathbf{x}$ | order does not matter |
| distributes over $+$ | $\mathbf{x}\cdot(\mathbf{y}+\mathbf{z}) = \mathbf{x}\cdot\mathbf{y} + \mathbf{x}\cdot\mathbf{z}$ | splits across sums |
| scalar scaling | $(c\mathbf{x})\cdot\mathbf{y} = c(\mathbf{x}\cdot\mathbf{y})$ | constants factor out |
| self dot | $\mathbf{x}\cdot\mathbf{x} = \sum_i x_i^2 \ge 0$ | equals $\|\mathbf{x}\|^2$, zero only for $\mathbf{0}$ |

Note what the self dot product gives: a sum of squares — always non-negative, and exactly zero only when every component is zero. That property makes $\sqrt{\mathbf{x}\cdot\mathbf{x}}$ a legitimate notion of length.

## Geometric Meaning: Angle

The dot product connects algebra to geometry through the law of cosines (section 00). Two steps:

**Step 1 — expand algebraically.** For $\mathbf{x}, \mathbf{y}$:

$$
\|\mathbf{x} - \mathbf{y}\|^2 = (\mathbf{x}-\mathbf{y})\cdot(\mathbf{x}-\mathbf{y})
= \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\,\mathbf{x}\cdot\mathbf{y}
$$

(using distributivity, and $\mathbf{x}\cdot\mathbf{y} = \mathbf{y}\cdot\mathbf{x}$).

**Step 2 — the same quantity geometrically.** The law of cosines, with $\theta$ the angle between the two arrows:

$$
\|\mathbf{x}-\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta
$$

Comparing the two expressions, the $\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2$ terms cancel:

$$
\boxed{\;\mathbf{x}\cdot\mathbf{y} = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta\;}
$$

This identity is not an extra axiom — it follows from the algebraic definition. Consequences:

- **$\mathbf{x}\cdot\mathbf{y} = 0 \iff \theta = 90°$** (assuming both vectors non-zero). Zero dot product = perpendicular = **orthogonal**.
- The sign of $\mathbf{x}\cdot\mathbf{y}$ tells you the direction: positive for acute angles, negative for obtuse.
- If both vectors are non-negative (like probabilities or feature counts), the dot product can never be negative — worth noticing when it appears in attention.

## Norms (Length)

A **norm** is a function $\|\cdot\|$ assigning a non-negative size to each vector, with:

1. $\|\mathbf{x}\| \ge 0$, and $= 0$ only for $\mathbf{x} = \mathbf{0}$,
2. $\|c\mathbf{x}\| = |c|\,\|\mathbf{x}\|$ (scaling stretches size proportionally),
3. $\|\mathbf{x}+\mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$ (triangle inequality: a detour never beats a straight line).

### Common norms

**Euclidean norm ($L^2$)** — ordinary length:

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2} = \sqrt{\mathbf{x}\cdot\mathbf{x}}
$$

**Manhattan norm ($L^1$):**

$$
\|\mathbf{x}\|_1 = \sum_i |x_i|
$$

**General $L^p$ norm:**

$$
\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}
$$

**Worked example.** $\mathbf{x} = (3, -4)$:

$$
\|\mathbf{x}\|_1 = 3 + 4 = 7, \qquad
\|\mathbf{x}\|_2 = \sqrt{9 + 16} = \sqrt{25} = 5
$$

```python
def norm2(x):
    return sum(v * v for v in x) ** 0.5
```

Why $L^1$ and $L^2$ both appear in AI: $L^2$ measures straight-line error (regression loss, weight decay); $L^1$ counts absolute error and produces sparse solutions (lasso). The choice of norm changes what a model prefers — section 05 returns to this.

## Distance

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2
= \sqrt{\sum_i (x_i - y_i)^2}
$$

The distance from $\mathbf{x}$ to $\mathbf{y}$ is the norm of their difference — nothing new to memorize. $d$ is symmetric ($d(\mathbf{x},\mathbf{y}) = d(\mathbf{y},\mathbf{x})$), zero only for identical vectors, and satisfies the triangle inequality: $d(\mathbf{x},\mathbf{z}) \le d(\mathbf{x},\mathbf{y}) + d(\mathbf{y},\mathbf{z})$.

**Worked example.** $\mathbf{x} = (1, 2)$, $\mathbf{y} = (4, 6)$:

$$
d = \sqrt{(4-1)^2 + (6-2)^2} = \sqrt{9 + 16} = 5
$$

## Cosine Similarity

Length and direction carry different information. Two product reviews may be "similar" in topic regardless of length. **Cosine similarity** compares direction only:

$$
\cos\theta = \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|}
$$

Range: $[-1, 1]$. Value $1$ = same direction, $0$ = orthogonal (unrelated), $-1$ = opposite direction.

**Worked example.**

$$
\mathbf{x} = (1,2,3), \quad \mathbf{y} = (4,5,6):
\quad \|\mathbf{x}\| = \sqrt{14},\;\; \|\mathbf{y}\| = \sqrt{77}
$$

$$
\cos\theta = \frac{32}{\sqrt{14}\sqrt{77}} = \frac{32}{\sqrt{1078}} \approx \frac{32}{32.83} \approx 0.975
$$

$\theta \approx 12.9°$ — nearly the same direction, though $\|\mathbf{x}\| \approx 3.74$ and $\|\mathbf{y}\| \approx 8.77$ differ by more than a factor of two. Direction similar, magnitude different: exactly the distinction cosine similarity isolates.

The vector $(0,0)$ has no direction — cosine similarity is undefined there. Real implementations add a tiny $\varepsilon$ to the denominator to avoid division by zero.

## Why the Dot Product Works as a Score

Rewrite the geometric identity:

$$
\frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|} = \cos\theta
$$

The dot product alone grows with length: a long vector gets big scores with everything. Dividing by the lengths removes that bias. Three regimes:

| score | behavior |
|---|---|
| raw dot product $\mathbf{x}\cdot\mathbf{y}$ | favors long vectors — a length-weighted similarity |
| cosine similarity | compares direction only — scale-invariant |
| both normalized to length 1 | identical: $\cos\theta = \mathbf{x}\cdot\mathbf{y}$ |

When every vector is pre-normalized to unit length, the raw dot product *is* cosine similarity — an implementation shortcut used in embedding search.

## Connection to AI

- **Similarity search.** Embedding systems retrieve the item with the highest cosine similarity to a query vector. This chapter's formula is the search engine's core computation.
- **Attention scores.** $\operatorname{Attention}$ computes $QK^\top$ — every entry a dot product between a query and a key (section 09). The scaling by $1/\sqrt{d_k}$ exists because raw dot products grow with dimension; you will be able to derive that in section 09 from the variance of a sum of products.
- **Linear classifiers.** $\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$: the whole prediction is one dot product plus a bias (section 07).
- **Regularization.** Weight decay penalizes $\|\mathbf{w}\|_2^2 = \mathbf{w}\cdot\mathbf{w}$ — the self dot product (section 05).
- **Losses.** Mean squared error is $\frac{1}{n}\sum (y_i - \hat y_i)^2$, built from $L^2$ distances.

## Common Mistakes

- Taking a dot product of vectors with different lengths — the sum $\sum_i x_iy_i$ requires matching indices.
- Confusing $\mathbf{x}\cdot\mathbf{y}$ (scalar) with $\mathbf{x}\mathbf{y}^\top$ (an $n\times n$ matrix — outer product, chapter 03).
- Assuming large dot product = similar. Without normalization, it may only mean large norms.
- Forgetting that $\|\mathbf{x}\|^2 = \mathbf{x}\cdot\mathbf{x}$, not $\mathbf{x}\cdot\mathbf{x}$ squared again.
- Reading cosine similarity $0$ as "no relation" when the vectors merely are orthogonal — uncorrelated, not causally independent (section 03 distinguishes these).

## Exercises

1. Compute $\mathbf{x}\cdot\mathbf{y}$ for $\mathbf{x} = (2, -1, 5)$, $\mathbf{y} = (3, 4, -2)$.
2. Compute $\|\mathbf{x}\|_2$ and $\|\mathbf{x}\|_1$ for $\mathbf{x} = (3, -4, 0)$.
3. Are $\mathbf{u} = (1, 2)$ and $\mathbf{v} = (2, -1)$ orthogonal? Show your reasoning with the dot product.
4. Compute the distance between $(1, 1)$ and $(4, 5)$.
5. Compute the cosine similarity of $\mathbf{x} = (1, 0)$ and $\mathbf{y} = (1, 1)$; then give the angle in degrees.
6. Prove from the definition that $\mathbf{0} \cdot \mathbf{y} = \mathbf{0}$ for any $\mathbf{y}$, and state why this is consistent with the geometric formula.
7. Write a Python function `cosine(x, y)` using only `sum` and `** 0.5`.
