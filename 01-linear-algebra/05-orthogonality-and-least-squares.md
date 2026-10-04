# Orthogonality and Least Squares

## Why It Matters for AI

Data rarely obeys a model exactly. Fitting a line to scattered points means finding the closest possible model — and "closest" is defined with norms and dot products. The **least squares** method turns a fitting problem into a linear system, solving it in closed form with the transpose. This chapter completes the first chain of the section:

```text
Vector → Dot Product → Projection → Orthogonality → Least Squares
```

Linear regression (section 07) is this chain applied to prediction.

## Orthogonality

Vectors $\mathbf{x}, \mathbf{y}$ are **orthogonal** when $\mathbf{x} \cdot \mathbf{y} = 0$ (for non-zero vectors: the angle is $90°$, chapter 02). Write $\mathbf{x} \perp \mathbf{y}$.

An **orthogonal set** has every distinct pair at $90°$. The standard basis $\mathbf{e}_1, \ldots, \mathbf{e}_n$ is orthogonal; if additionally each vector has unit length, the set is **orthonormal**:

$$
\mathbf{e}_i \cdot \mathbf{e}_j = \begin{cases} 1 & i = j \\ 0 & i \ne j \end{cases}
$$

Why orthogonal bases are preferred: coordinates are trivial to read. For $\mathbf{x} = c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k$ with $\{\mathbf{v}_i\}$ orthogonal, dotting both sides with $\mathbf{v}_j$ kills every term except $j$:

$$
c_j = \frac{\mathbf{x} \cdot \mathbf{v}_j}{\mathbf{v}_j \cdot \mathbf{v}_j}
\qquad(\text{for orthonormal: } c_j = \mathbf{x} \cdot \mathbf{v}_j)
$$

No linear system to solve — the coefficients fall out of $k$ dot products. (Gram–Schmidt is the algorithm that converts any basis into an orthogonal one; not needed here, but now you know what it is for.)

## Projection onto a Line

**Problem.** Given $\mathbf{x}$ and a direction $\mathbf{u} \ne \mathbf{0}$, find the point $\mathbf{p}$ on the line through $\mathbf{u}$ closest to $\mathbf{x}$.

The closest point must make the residual $\mathbf{x} - \mathbf{p}$ **perpendicular** to the line. If it were not, you could slide $\mathbf{p}$ along the line and get closer — the residual would have a component along the line to cancel. So the condition is:

$$
(\mathbf{x} - c\mathbf{u}) \cdot \mathbf{u} = 0
$$

Solving for $c$:

$$
\mathbf{x}\cdot\mathbf{u} - c(\mathbf{u}\cdot\mathbf{u}) = 0
\quad\Rightarrow\quad
c = \frac{\mathbf{x}\cdot\mathbf{u}}{\mathbf{u}\cdot\mathbf{u}}
$$

$$
\boxed{\;\operatorname{proj}_{\mathbf{u}}\mathbf{x} = \frac{\mathbf{x}\cdot\mathbf{u}}{\mathbf{u}\cdot\mathbf{u}}\,\mathbf{u}\;}
$$

**Worked example.** Project $\mathbf{x} = (3, 1)$ onto $\mathbf{u} = (1, 1)$ (the line $y = x$):

$$
c = \frac{3\cdot1 + 1\cdot1}{1\cdot1 + 1\cdot1} = \frac{4}{2} = 2,
\qquad
\mathbf{p} = 2(1,1) = (2,2)
$$

Residual: $\mathbf{x} - \mathbf{p} = (1, -1)$. Check orthogonality: $(1,-1)\cdot(1,1) = 1 - 1 = 0$ ✓

Distance from $\mathbf{x}$ to the line: $\|(1,-1)\| = \sqrt{2} \approx 1.414$ — smaller than to any other point on the line.

```python
def project(x, u):
    c = dot(x, u) / dot(u, u)
    return [c * ui for ui in u]
```

## Projection onto a Subspace

For a subspace spanned by the columns of $A$ (an $m \times n$ matrix), the projection of $\mathbf{b} \in \mathbb{R}^m$ is:

$$
\mathbf{p} = A(A^\top A)^{-1}A^\top \mathbf{b}
$$

and the whole operation is one matrix:

$$
P = A(A^\top A)^{-1}A^\top
$$

$P$ is the **projection matrix**: $P^2 = P$ (projecting twice changes nothing), $P^\top = P$ (symmetric). The formula generalizes the line case: for $A = \mathbf{u}$ it reduces to $\frac{\mathbf{u}\mathbf{u}^\top}{\mathbf{u}^\top\mathbf{u}}$ applied to $\mathbf{x}$.

Requirements: the columns of $A$ must be independent so $A^\top A$ is invertible (if they are dependent, remove the redundant columns first — rank from chapter 04).

## The Least Squares Problem

**Setting.** Predict $y$ from $x$ with a straight line. Take three data points and fit $y = bx + w$... using the standard form $\hat{y} = b + wx$ where $b$ is the intercept and $w$ the slope:

| $x$ | $y$ |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 5 |

Model: $\hat{y}_i = b + w x_i$. In matrix form, with the design matrix $X$ (a column of 1s for the intercept, a column of $x$ values):

$$
X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad
\mathbf{w} = \begin{bmatrix} b \\ w \end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix},
\qquad
X\mathbf{w} \approx \mathbf{y}
$$

Three equations, two unknowns — generally no exact solution (the points are not exactly collinear). **Least squares** chooses $\mathbf{w}$ to minimize the sum of squared vertical errors:

$$
J(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|_2^2 = \sum_{i=1}^{3} (y_i - \hat{y}_i)^2
$$

## Deriving the Normal Equations

Key insight: $\hat{\mathbf{y}} = X\mathbf{w}$ lies in the column space of $X$ (a linear combination of columns — chapter 03). The best approximation to $\mathbf{y}$ from that subspace is the **projection** of $\mathbf{y}$ onto it. By the projection argument above, the residual must be orthogonal to every column of $X$:

$$
\mathbf{y} - X\mathbf{w} \perp \text{columns of } X
$$

Writing "$\perp$ columns" compactly as $X^\top(\mathbf{y} - X\mathbf{w}) = \mathbf{0}$ (each entry is one dot product with a column):

$$
X^\top X \mathbf{w} = X^\top \mathbf{y}
$$

$$
\boxed{\;\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}\;}
$$

These are the **normal equations** — the closed-form solution of linear regression. (Section 05 will show the same result by setting the gradient to zero; the two routes agree because "orthogonal residual" and "zero gradient" say the same thing.)

The interpretation of each piece:

| piece | role |
|---|---|
| $X^\top X$ | $n \times n$ symmetric matrix of feature self-products |
| $X^\top \mathbf{y}$ | correlations between features and target |
| $(X^\top X)^{-1}$ | undoes the mixing (requires independent columns) |
| $X^\top X \mathbf{w} = X^\top \mathbf{y}$ | each feature's residual correlation must vanish |

## Worked Example (the three points)

$$
X^\top X = \begin{bmatrix} 1&1&1\\1&2&3 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}
= \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}
$$

$$
X^\top \mathbf{y} = \begin{bmatrix} 1&1&1\\1&2&3 \end{bmatrix}
\begin{bmatrix} 2\\3\\5 \end{bmatrix}
= \begin{bmatrix} 2+3+5 \\ 2+6+15 \end{bmatrix}
= \begin{bmatrix} 10 \\ 23 \end{bmatrix}
$$

Invert the $2 \times 2$: $\det = 3 \cdot 14 - 6 \cdot 6 = 6$, so

$$
(X^\top X)^{-1} = \frac{1}{6}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}
$$

$$
\mathbf{w} = \frac{1}{6}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}\begin{bmatrix} 10 \\ 23 \end{bmatrix}
= \frac{1}{6}\begin{bmatrix} 140 - 138 \\ -60 + 69 \end{bmatrix}
= \frac{1}{6}\begin{bmatrix} 2 \\ 9 \end{bmatrix}
= \begin{bmatrix} 1/3 \\ 3/2 \end{bmatrix}
$$

Best line: $\hat{y} = \tfrac{1}{3} + \tfrac{3}{2}x$ (intercept $b = 1/3$, slope $w = 3/2$).

Predictions and residuals:

| $x$ | $y$ | $\hat y$ | residual $y - \hat y$ |
|---|---|---|---|
| 1 | 2 | $1/3 + 1.5 = 1.833$ | $+0.167$ |
| 2 | 3 | $1/3 + 3 = 3.333$ | $-0.333$ |
| 3 | 5 | $1/3 + 4.5 = 4.833$ | $+0.167$ |

Verify orthogonality: residuals sum to zero (dot product with the 1-column) $0.167 - 0.333 + 0.167 \approx 0$ ✓, and dot product with the $x$-column: $1(0.167) + 2(-0.333) + 3(0.167) = 0.167 - 0.667 + 0.500 = 0$ ✓. Both defining conditions hold.

Sum of squared errors: $\left(\tfrac{1}{6}\right)^2 + \left(\tfrac{1}{3}\right)^2 + \left(\tfrac{1}{6}\right)^2 = \tfrac{1}{36} + \tfrac{4}{36} + \tfrac{1}{36} = \tfrac{1}{6} \approx 0.167$ — the minimum achievable by any line.

## Why Orthogonality = Optimality

The chain of reasoning, stated once more because it recurs throughout the course:

1. Minimizing $\|\mathbf{y} - \hat{\mathbf{y}}\|^2$ over a subspace means finding the closest point in that subspace.
2. The closest point makes the residual orthogonal to the subspace (any non-orthogonal component could be cancelled to get closer).
3. "Orthogonal to all columns of $X$" is the linear system $X^\top X\mathbf{w} = X^\top \mathbf{y}$.

Geometry (right angle) and optimization (minimum) are two views of the same condition. In section 05 you will see the third view: $\nabla J = 0$.

## Connection to AI

- **Linear regression** is exactly this derivation (section 07 adds the general matrix case and gradient descent when $X^\top X$ is huge or ill-conditioned).
- **Projection = dimensionality reduction in disguise.** Replacing a vector by its projection onto a smaller subspace is lossy compression — PCA (section 07) chooses the subspace that loses least in a variance sense.
- **Residuals orthogonal to features** is the defining property of a least-squares fit; it is also why no *linear* pattern remains in the residuals of a correct fit.
- **Gram matrices $X^\top X$ appear everywhere**: covariance matrices (section 04), the normal equations, and $K = X X^\top$ as a similarity matrix between examples.
- **Attention's $QK^\top$** is a matrix of dot products — with orthonormalized features it measures angle, as in this chapter (section 09).

## Common Mistakes

- Dropping the inverse in $P = A(A^\top A)^{-1}A^\top$ — $A A^\top$ is not the projection (wrong shape for most cases and wrong matrix in general).
- Reversing the product: $A^\top A \ne AA^\top$ in general; dimensions differ.
- Assuming least squares minimizes absolute errors — it minimizes **squared** errors; $L^1$ fitting is a different method (median-based).
- Forgetting the intercept: fitting $y = wx$ through the origin (no 1-column) gives different coefficients than $y = b + wx$.
- Interpreting orthogonality of residuals as "the model is correct" — only that no *further linear* improvement is available from those features.

## Exercises

1. Project $\mathbf{x} = (4, 2)$ onto $\mathbf{u} = (1, 0)$; verify the residual is orthogonal to $\mathbf{u}$.
2. Project $\mathbf{x} = (2, 3)$ onto the line spanned by $(1, 1)$. What is the distance from $\mathbf{x}$ to the line?
3. Show that if $\mathbf{x}$ is already on the line through $\mathbf{u}$, then $\operatorname{proj}_\mathbf{u} \mathbf{x} = \mathbf{x}$.
4. Fit $y = b + wx$ by normal equations to the two points $(1, 3)$ and $(2, 5)$. (Hint: two points determine a line — the residuals will be zero.)
5. For $A = \begin{bmatrix}1\\1\end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix}1\\3\end{bmatrix}$: compute $A^\top A$, $A^\top \mathbf{b}$, the least squares solution, and the residual. Is the residual orthogonal to the column of $A$?
6. Instead of the fitted line, consider the best line **through the origin** $\hat y = wx$ on the same three points: $w = \frac{\sum x_iy_i}{\sum x_i^2} = 23/14$. Compute its residual sum of squares and compare with $1/6$ from the fitted line with intercept. What does the comparison tell you about constraining the model?
