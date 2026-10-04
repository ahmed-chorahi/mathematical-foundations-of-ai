# Linear Regression

## Why It Matters for AI

Linear regression is the model where every thread of this course comes together visibly: squared error from Gaussian likelihood (section 04), the normal equations as a projection onto a subspace (section 01), convexity of the objective (section 05), and training by gradient descent (section 05, chapter 02). It has a closed-form solution — the only mainstream model that does — which makes it the place to see *exactly* what "fitting a model" means.

## The Model

Data: $n$ examples with features $x_i \in \mathbb{R}$ (extendable to $d$ features) and targets $y_i \in \mathbb{R}$.

$$
\hat y = wx + b \qquad\text{(one feature)}
$$

Matrix form with design matrix $X \in \mathbb{R}^{n \times (d+1)}$ (first column all ones — the intercept column) and $\theta \in \mathbb{R}^{d+1}$:

$$
\hat{\mathbf{y}} = X\theta
$$

**Why squared error:** with Gaussian noise $y_i = \theta^\top x_i + \varepsilon$, $\varepsilon \sim N(0, \sigma^2)$ (section 04), maximizing likelihood over $\theta$ = minimizing $\sum_i (y_i - \hat y_i)^2$. The loss is not arbitrary — it is the MLE under an assumption.

$$
J(\theta) = \frac{1}{n}\|X\theta - \mathbf{y}\|^2
$$

## The Normal Equations (Closed Form)

From section 05's convexity chapter: $J$ is convex with Hessian $2X^\top X \succeq 0$. Set gradient to zero:

$$
\nabla J = \frac{2}{n} X^\top(X\theta - \mathbf{y}) = \mathbf{0}
\;\;\Longrightarrow\;\;
X^\top X\,\theta = X^\top \mathbf{y}
$$

$$
\boxed{\;\theta^* = (X^\top X)^{-1} X^\top \mathbf{y}\;}
$$

provided $X^\top X$ is invertible (columns of $X$ linearly independent — section 01: full column rank). Interpretation from section 01: $\hat{\mathbf{y}} = X\theta^*$ is the **projection of $\mathbf{y}$ onto the column space of $X$** — the closest reachable output; $X^\top(X\theta - \mathbf{y}) = \mathbf{0}$ says the residual is orthogonal to every column (the normal equations are literally "perpendicular").

If $X^\top X$ is singular (duplicate/collinear features): use the pseudoinverse for the minimum-norm solution, or add $\lambda I$ — **ridge**, section 05's closed form $ (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$.

**Worked example.** Points $(x, y)$: $(1, 2),\ (2, 4),\ (3, 5)$. Fit $\hat y = a + bx$ ($X$ has the ones column):

$$
X^\top X = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix},
\qquad
X^\top \mathbf{y} = \begin{bmatrix} 11 \\ 25 \end{bmatrix}
\quad (n = 3,\ \textstyle\sum x = 6,\ \sum x^2 = 14,\ \sum y = 11,\ \sum xy = 25)
$$

Invert the $2 \times 2$ (determinant $3\cdot14 - 36 = 6$):

$$
(X^\top X)^{-1} = \frac{1}{6}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix},
\qquad
\theta^* = \frac{1}{6}\begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}\begin{bmatrix} 11 \\ 25 \end{bmatrix}
= \frac{1}{6}\begin{bmatrix} 4 \\ 9 \end{bmatrix}
= \begin{bmatrix} 0.667 \\ 1.5 \end{bmatrix}
$$

**$\hat y = 0.667 + 1.5x$**, fitted values $2.167,\ 3.667,\ 5.167$, residuals $-0.167,\ 0.333,\ -0.167$, SSE $\approx 0.167$ (verified numerically; residuals sum to $0$ — the ones-column condition).

```python
def solve2(A, v):                       # A: 2x2, v: 2
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    return [( A[1][1]*v[0] - A[0][1]*v[1]) / det,
            (-A[1][0]*v[0] + A[0][0]*v[1]) / det]
# X^T X = [[3,6],[6,14]], X^T y = [11,25]
print(solve2([[3,6],[6,14]], [11,25]))   # [0.6667, 1.5]
```

## Fitting by Gradient Descent

The closed form needs a matrix inverse — $O(d^3)$, awkward for large $d$. Gradient descent needs only multiplications:

$$
\theta \leftarrow \theta - \eta \cdot \frac{2}{n} X^\top(X\theta - \mathbf{y})
$$

One step from $\theta = (0, 0)$ on the worked example ($\eta = 0.01$):

$$
X\theta - \mathbf{y} = -\mathbf{y} = (-2, -4, -5),\quad
X^\top(\text{residual}) = \begin{bmatrix} 11 \\ 25 \end{bmatrix},\quad
\theta_1 = 0 + 0.01 \cdot 2 \cdot \frac{1}{3}\begin{bmatrix} 11 \\ 25 \end{bmatrix} = \begin{bmatrix} 0.0733 \\ 0.1667 \end{bmatrix}
$$

Both parameters grow toward the solution; repeated steps converge (the objective is convex — chapter 03 of section 05 — so the limit is the global optimum, and for quadratics the rate depends on the eigenvalue spread of $X^\top X$: ill-conditioned features converge slowly — another reason to standardize, section 04).

**Closed form vs gradient descent:** exact answer vs scalable computation; the same minimum. Production systems use GD-family optimizers because $d$ (and $n$) grow past what inverses handle.

## Reading the Fit

- **Residuals** $r_i = y_i - \hat y_i$: the part the model didn't reach (the projection's orthogonal complement).
- **Goodness of fit:** $R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$ — fraction of target variance explained ($\mathrm{SST} = \sum(y_i - \bar y)^2$). $R^2 = 1$ perfect; $0$ no better than the mean.
- **Parameters:** $b$ = change in $y$ per unit $x$ (holding others fixed) — readable, which is linear regression's enduring strength.

## Extension to Multiple Features

Nothing changes: $X$ gains columns; $X^\top X$ becomes $(d+1)\times(d+1)$; the derivation is identical. Polynomial regression ($x \to [1, x, x^2]$) is *still linear regression** — linear in the parameters, not the inputs — and inherits every result here (including overfitting with high degree — section 05's regularization applies).

## Connection to AI

- **The template model:** every later model replaces $\hat y = \theta^\top x$ with something richer, keeping the skeleton: define loss → differentiate → iterate (or solve).
- **Neural networks** with no activation are linear regression; the first layer of every network *is* a regression on learned features (section 08).
- **Meaning of "loss going down":** here you can see the target — $\theta$ walking to the exact minimizer. Later sections lose the closed form but not the logic.
- **Feature engineering's cost:** collinearity makes $X^\top X$ ill-conditioned (slow GD, unstable coefficients) — preprocessing (section 04) is not optional hygiene.
- **GLMs:** logistic regression (next chapter) keeps $\eta = \theta^\top x$ and changes only the link/loss — linear regression is the canonical generalized linear model.

## Common Mistakes

- Inverting $X^\top X$ when it is singular — check rank (section 01) or use ridge/pseudoinverse.
- Forgetting the intercept column (or not centering) — then the "no-feature" baseline is wrong and coefficients get pulled.
- Interpreting fit as causation — $\theta$ describes association in the data; no intervention claim.
- Comparing $R^2$ across different targets — it scales with the target's variance.
- Assuming Gaussian noise for *validity of the coefficients' algebra* — the projection solution needs only least squares; Gaussian noise is what makes it MLE and gives standard errors.
- Confusing $O(d^3)$ inverse cost with GD's per-step $O(nd)$ — the reason large-scale fitting never inverts anything.

## Exercises

1. Derive the normal equations from $J(\theta) = \|X\theta - \mathbf{y}\|^2$ (start from the gradient; state the condition for existence).
2. Fit $\hat y = a + bx$ to $(1, 1), (2, 3)$: compute $X^\top X$, $X^\top \mathbf{y}$, solve; predict $x = 3$.
3. For that fit, compute SSE. What is $R^2$ if $\mathrm{SST} = 2$? ($\bar y = 2$.)
4. Take one GD step from $\theta = (0,0)$ on the chapter's example with $\eta = 0.1$ (not $0.01$) — compare with the chapter's $\eta = 0.01$ step.
5. When is $X^\top X$ singular? Give one concrete 3-row design matrix (with intercept column) that causes it.
6. Why is polynomial regression still "linear regression"? State what "linear" refers to.
