# 07 — Machine Learning Mathematics: Solutions

## 01-linear-regression.md

**1. Derivation of the normal equations.**
$$
\nabla J(\theta) = \nabla\big(\theta^\top X^\top X \theta - 2\theta^\top X^\top \mathbf{y} + \mathbf{y}^\top\mathbf{y}\big) = 2X^\top X\theta - 2X^\top \mathbf{y} = \mathbf{0}
$$
$$
X^\top X\,\theta = X^\top \mathbf{y}
$$
Exists uniquely iff $X^\top X$ is invertible, i.e. $X$ has full column rank (its columns linearly independent — section 01); otherwise use the pseudoinverse or ridge.

**2. Fit to $(1,1), (2,3)$.** $X^\top X = \begin{bmatrix}2&3\\3&5\end{bmatrix}$, $X^\top\mathbf{y} = \begin{bmatrix}4\\7\end{bmatrix}$ (from $n=2$, $\sum x = 3$, $\sum x^2 = 5$, $\sum y = 4$, $\sum xy = 7$). Determinant $= 1$; inverse $\begin{bmatrix}5&-3\\-3&2\end{bmatrix}$:

$$
\theta = \begin{bmatrix}5&-3\\-3&2\end{bmatrix}\begin{bmatrix}4\\7\end{bmatrix} = \begin{bmatrix}-1\\2\end{bmatrix}
\;\Longrightarrow\; \hat y = -1 + 2x,\qquad \hat y(3) = 5
$$

(Exact fit through two points: residuals are $0, 0$.)

**3. SSE and $R^2$.** Residuals $0, 0$ ⇒ $\mathrm{SSE} = 0$; $R^2 = 1 - 0/\mathrm{SST} = 1 - 0/2 = 1$ — a perfect fit (expected: two points, two parameters).

**4. One GD step with $\eta = 0.1$.** Same computation as the chapter with $10\times$ the step:
$$
\theta_1 = 0.1 \cdot \tfrac{2}{3} \begin{bmatrix}11\\25\end{bmatrix} = \begin{bmatrix}0.7333\\1.6667\end{bmatrix}
$$
vs the chapter's $\eta = 0.01$ step $(0.0733, 0.1667)$ — ten times larger, and already slightly *past* the optimum $(0.667, 1.5)$ (overshoot — chapter 02 of section 05's learning-rate story).

**5. When is $X^\top X$ singular — example.** When the columns of $X$ are linearly dependent (collinear/duplicate features, or fewer rows than parameters). Example (intercept + two features where $x_3 = 2x_2$):
$$
X = \begin{bmatrix}1&1&2\\1&2&4\\1&3&6\end{bmatrix}
\;\Longrightarrow\; \text{column 3} = 2\times\text{column 2} \;\Longrightarrow\; X^\top X \text{ singular}
$$

**6. What "linear" means.** Linear in the **parameters** $\theta$: the model is $\hat y = \sum_j \theta_j \phi_j(x)$ with fixed basis functions $\phi_j$; $J(\theta)$ is quadratic in $\theta$ and the normal equations apply. Features may be nonlinear in $x$ ($x^2$, $\sin x$, interactions) — still linear regression.

## 02-logistic-regression.md

**1. Sigmoid values.** $\sigma(0) = 0.5$; $\sigma(2) = 0.8808$; $\sigma(-2) = 0.1192$. Sum: $0.8808 + 0.1192 = 1$ ✓ (symmetry $\sigma(-z) = 1 - \sigma(z)$).

**2. $w = (1,1)$, $b = -1.5$, $x = (1,0)$.** $z = 1 - 1.5 = -0.5 < 0$ — below the boundary ⇒ predict class 0; probability $q = \sigma(-0.5) \approx 0.3775$.

**3. Derivative of $\ln(1 + e^{-z})$.** Chain rule with $u = 1 + e^{-z}$, $du/dz = -e^{-z}$:
$$
\frac{d}{dz}\ln u = \frac{-e^{-z}}{1 + e^{-z}} = \frac{-1}{e^{z} + 1} = -\sigma(-z) = \sigma(z) - 1 \quad\text{(using } \sigma(z) + \sigma(-z) = 1\text{)} \;✓
$$
i.e., the $y = 1$ case of $\partial\ell/\partial z = \sigma(z) - y$.

**4. GD step, $y = 0$, $x = (2,3)$, $\theta = 0$, $\eta = 0.1$.** $z = 0$, $\sigma(z) = 0.5$, error $= 0.5 - 0 = 0.5$; $\nabla_w \ell = 0.5 \cdot (2, 3) = (1, 1.5)$, $\partial\ell/\partial b = 0.5$:
$$
w \leftarrow (0,0) - 0.1(1, 1.5) = (-0.1, -0.15),\qquad b \leftarrow -0.05
$$
(For this same example $x = (2,3)$, the new $z = (-0.1)(2) + (-0.15)(3) - 0.05 = -0.7$, so $q = \sigma(-0.7) \approx 0.3318$ — moved down from $0.5$ toward the true label $0$ ✓.)

**5. $y = 1$, $z = -3$.** $\ell = \ln(1 + e^{3}) \approx 3.0486$ nats — huge compared with the confident-correct case ($z = +3$: $\ln(1 + e^{-3}) \approx 0.0486$): cross-entropy charges roughly linearly-in-$|z|$ for confident errors, which is why separable data drives $\|w\| \to \infty$ (likelihood keeps improving).

**6. Why convex.** $\ell(z)$ is convex in the logit (softplus) and $z = w^\top x + b$ is affine; convex-composed-with-affine stays convex (section 05's composition rule). Deep nets compose *nonlinear* layers, leaving the convex class — hence non-convexity.

## 03-pca.md

**1. Derivation.** Maximize $u^\top\Sigma u$ subject to $u^\top u = 1$; Lagrangian $\mathcal{L} = u^\top \Sigma u - \lambda(u^\top u - 1)$; stationarity: $\nabla_u \mathcal{L} = 2\Sigma u - 2\lambda u = \mathbf{0} \Rightarrow \Sigma u = \lambda u$ — the optimizer is an eigenvector, with attained variance $u^\top\Sigma u = \lambda$.

**2. $\Sigma = \mathrm{diag}(5, 2)$.** Eigenvalues $5$ and $2$ (diagonal entries), eigenvectors $e_1 = (1,0)$, $e_2 = (0,1)$. PC1 ratio: $5/(5+2) = 5/7 \approx 0.714$ (71.4% of variance).

**3. Direct check of $u_1^\top \Sigma u_1$.**
$$
u_1^\top \Sigma u_1 = \tfrac12 (1, 1)\begin{bmatrix}2&1.6\\1.6&2\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}
= \tfrac12 (1,1)\begin{bmatrix}3.6\\3.6\end{bmatrix} = \tfrac12 (7.2) = 3.6 \;✓
$$

**4. Why scores are uncorrelated.** $\operatorname{Cov}(z_j, z_l) = u_j^\top \Sigma u_l = \lambda_l\, u_j^\top u_l = 0$, because eigenvectors of the symmetric $\Sigma$ are mutually orthogonal (spectral theorem — section 01).

**5. Standardized $d = 5$.** Diagonal entries $= 1$ ⇒ $\operatorname{tr}(\Sigma) = 5$; since $\operatorname{tr} = \sum_i \lambda_i$ (trace invariant under the eigendecomposition — section 01), the eigenvalues sum to $5$ = total variance.

**6. Cost of dropping PC2.** Reconstruction MSE equals the dropped eigenvalues' sum $= \lambda_2 = 0.4$ — exactly the $10\%$ of total variance ($4.0$) PC2 carried: best rank-1 reconstruction error $=$ variance discarded.
