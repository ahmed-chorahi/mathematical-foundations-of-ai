# Matrices

## Why It Matters for AI

A matrix is how you store parameters and apply them. A neural-network layer holds a weight matrix $W$; multiplying $W$ by an input vector produces the next representation. Data itself is a matrix: one row per example, one column per feature. Matrix multiplication is therefore not bookkeeping — it *is* the computation a model performs billions of times per training run.

## Definition and Notation

A **matrix** $A$ with $m$ rows and $n$ columns ($m \times n$) is a rectangular array:

$$
A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}
$$

Entry in row $i$, column $j$: $a_{ij}$ (first index = row, second = column). $A \in \mathbb{R}^{m \times n}$ records the dimension. A vector is a matrix with one column ($n \times 1$); a scalar is a $1 \times 1$ matrix.

## Addition and Scalar Multiplication

Same shape, componentwise — exactly as for vectors:

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 0 \\ -1 & 1 \end{bmatrix}
= \begin{bmatrix} 6 & 2 \\ 2 & 5 \end{bmatrix},
\qquad
2\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}
$$

Both require identical dimensions.

## Matrix–Vector Multiplication

Two equivalent definitions — the source of most intuition about matrices:

**Rows as dot products.** $(A\mathbf{x})_i = \sum_j a_{ij}x_j$ — the $i$-th output component is the dot product of row $i$ with $\mathbf{x}$.

**Columns as linear combinations.** $A\mathbf{x} = x_1\mathbf{c}_1 + x_2\mathbf{c}_2 + \cdots + x_n\mathbf{c}_n$, where $\mathbf{c}_j$ is column $j$ of $A$.

**Worked example.**

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\;\Rightarrow\;
A\mathbf{x} = 1\begin{bmatrix}1\\3\end{bmatrix} + 1\begin{bmatrix}2\\4\end{bmatrix}
= \begin{bmatrix} 3 \\ 7 \end{bmatrix}
$$

Check with rows: $(1\cdot1 + 2\cdot1,\; 3\cdot1 + 4\cdot1) = (3, 7)$ ✓

**Dimension rule:** $(m \times n) \cdot (n \times 1) \to (m \times 1)$. Inner dimensions must match: an $n$-component input is consumed by the $n$ columns of $A$.

## Matrix–Matrix Multiplication

$$
(AB)_{ik} = \sum_{j=1}^{n} A_{ij} B_{jk}
$$

Row $i$ of $A$ dotted with column $k$ of $B$. $A$ is $m \times n$, $B$ is $n \times p$, result $C = AB$ is $m \times p$.

**Worked example.**

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

$$
AB = \begin{bmatrix} 1\cdot0 + 2\cdot1 & 1\cdot1 + 2\cdot0 \\ 3\cdot0 + 4\cdot1 & 3\cdot1 + 4\cdot0 \end{bmatrix}
= \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}
$$

Now the reverse order:

$$
BA = \begin{bmatrix} 0\cdot1 + 1\cdot3 & 0\cdot2 + 1\cdot4 \\ 1\cdot1 + 0\cdot3 & 1\cdot2 + 0\cdot4 \end{bmatrix}
= \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}
$$

$AB \ne BA$. **Matrix multiplication is not commutative** — the order of layers in a network is therefore meaningful.

```python
def matmul(A, B):
    # A: m x n, B: n x p
    n = len(A[0])
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n))
             for j in range(p)]
            for i in range(len(A))]
```

Properties that *do* hold: **associative** $(AB)C = A(BC)$ — so layer groupings don't matter, only order; **distributive** $A(B+C) = AB + AC$.

## Transpose

Flip rows and columns: $(A^\top)_{ij} = A_{ji}$.

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}^\top
= \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix},
\qquad
\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}^\top
= \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
$$

Rules: $(A^\top)^\top = A$, $(A + B)^\top = A^\top + B^\top$, $(AB)^\top = B^\top A^\top$ (order reverses).

Note the third rule with $A = \mathbf{x}$ (a column) and $B = \mathbf{y}^\top$ (a row): $\mathbf{x}\mathbf{y}^\top$ is an $m \times n$ matrix (the **outer product**), while $\mathbf{y}^\top\mathbf{x}$ is a scalar (the dot product). Same vectors, different arrangement — a distinction worth internalizing before attention, where both appear.

## Identity Matrix

$I_n$ has 1s on the diagonal and 0s elsewhere:

$$
I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

$AI = IA = A$, and $I\mathbf{x} = \mathbf{x}$. The identity is to matrices what the number 1 is to multiplication — the neutral element.

## Inverse

$A^{-1}$ is the matrix with $A^{-1}A = AA^{-1} = I$. Then $A\mathbf{x} = \mathbf{b}$ can be solved by $\mathbf{x} = A^{-1}\mathbf{b}$.

For a $2 \times 2$ matrix, with **determinant** $\det(A) = ad - bc$:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \qquad
A^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
\quad\text{(requires } ad - bc \ne 0\text{)}
$$

**Worked example.** $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $\det = 1\cdot4 - 2\cdot3 = -2 \ne 0$:

$$
A^{-1} = \frac{1}{-2}\begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix}
= \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}
$$

Check: $\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}-2&1\\1.5&-0.5\end{bmatrix} = \begin{bmatrix}-2+3 & 1-1\\-6+6 & 3-2\end{bmatrix} = \begin{bmatrix}1&0\\0&1\end{bmatrix}$ ✓

When $\det = 0$ the inverse does not exist: the matrix is **singular**. Chapter 04 explains exactly what singularity means (some directions collapse to zero — the columns are redundant).

## Matrices as Linear Transformations

A matrix *is* a function. $T(\mathbf{x}) = A\mathbf{x}$ takes a vector in $\mathbb{R}^n$ to a vector in $\mathbb{R}^m$, and it preserves the linear structure:

$$
T(\mathbf{x} + \mathbf{y}) = T(\mathbf{x}) + T(\mathbf{y}), \qquad
T(c\mathbf{x}) = c\,T(\mathbf{x})
$$

(both are immediate from distributivity). A function with these two properties is a **linear transformation**.

Examples:

- $\begin{bmatrix}2&0\\0&2\end{bmatrix}$ scales the plane by 2.
- $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ rotates by 90° counterclockwise (check: it sends $(1,0)$ to $(0,1)$).
- $\begin{bmatrix}1&0\\0&0\end{bmatrix}$ flattens the plane onto the x-axis — the $y$-direction is lost, which is why this matrix has no inverse.

Two consequences that matter later:

1. **Composing transformations = matrix multiplication.** Apply $B$ then $A$: $\mathbf{x} \mapsto B\mathbf{x} \mapsto A(B\mathbf{x}) = (AB)\mathbf{x}$. Deep networks are long products $W_L \cdots W_2 W_1 \mathbf{x}$ — and multiplication's non-commutativity means the order of layers is part of the model's function.
2. **Every layer of a network is $A\mathbf{x}$ (plus a shift — chapter 07 treats $\mathbf{x} \mapsto W\mathbf{x} + \mathbf{b}$).**

## Data as a Matrix

$n$ examples, $d$ features each: one $n \times d$ matrix $X$.

$$
X = \begin{bmatrix} \text{example 1 features} \\ \text{example 2 features} \\ \vdots \end{bmatrix}
$$

The normal equations of regression (chapter 05), the covariance matrix (section 04), PCA (section 07), and attention ($QK^\top$) are all matrix products of $X$ with itself or with weight matrices.

## Connection to AI

- **Linear layers** $W\mathbf{x} + \mathbf{b}$: one matrix–vector product plus a shift, repeated dozens of times in a transformer block (section 09).
- **All parameters at once.** Forward passes over a batch of $n$ inputs $\mathbf{x}_1,\ldots,\mathbf{x}_n$ can be written $XW$ — one matrix product instead of $n$. Batching *is* matrix multiplication.
- **$Q = XW_Q$** (section 09): projecting every token embedding into queries is exactly a matrix–vector product applied to each row of the embedding matrix $X$.
- **Chain of layers = chain of products**, differentiated by the chain rule in section 08.

## Common Mistakes

- Multiplying incompatible shapes ($3\times2$ with $2\times3$ is fine; $3\times2$ with $3\times2$ is not) — always check inner dimensions match.
- Assuming $AB = BA$ or that $(AB)^{-1} = A^{-1}B^{-1}$ (it is $B^{-1}A^{-1}$).
- Forgetting $(AB)^\top = B^\top A^\top$ — the order flips.
- Confusing $\det(A) = 0$ meaning "no solutions" with "infinitely many": that question needs chapter 04.
- Treating division by a matrix as multiplication by $A^{-1}$ without checking invertibility exists.

## Exercises

1. Multiply: $\begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}\begin{bmatrix} 1 \\ 2 \end{bmatrix}$.
2. Compute $AB$ and $BA$ for $A = \begin{bmatrix}1&0\\0&2\end{bmatrix}$, $B = \begin{bmatrix}3&1\\0&1\end{bmatrix}$; are they equal?
3. Find the transpose of $\begin{bmatrix} 1 & -1 \\ 0 & 5 \end{bmatrix}$ and of the column vector $(1, 2)^\top$.
4. Find $A^{-1}$ for $A = \begin{bmatrix} 2 & 5 \\ 1 & 3 \end{bmatrix}$ and verify by matrix multiplication.
5. Show that $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ is singular (compute the determinant), and explain in one sentence what goes wrong in $A\mathbf{x}$.
6. Apply the rotation matrix $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ to $(1, 0)$ and describe the geometric result.
7. Derive $(AB)^\top = B^\top A^\top$ for the $2 \times 2$ case by computing both sides for $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B = \begin{bmatrix}0&1\\1&0\end{bmatrix}$.
