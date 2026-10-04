# Principal Component Analysis

## Why It Matters for AI

Data has redundant dimensions: height and weight correlate, pixel values in a patch co-vary, word counts across a corpus share structure. **PCA** finds the directions in which the data varies most, keeps the top few, and throws the rest away — linearly, optimally, and with a picture you can draw. It is the eigendecomposition of section 01 put to work on the covariance matrix of section 04, solving the constrained problem of section 05 — the cleanest example of "three sections combine into one algorithm".

## The Problem

Data $x_i \in \mathbb{R}^d$ (centered: subtract the mean first — section 04's standardization step; PCA is about *variation*, not location). Find a unit direction $u$ ($\|u\| = 1$) so that the projection of the data onto $u$ — the scores $z_i = u^\top x_i$ — has **maximum variance**:

$$
\max_{\|u\|=1} \operatorname{Var}(z) = \max_{\|u\|=1} u^\top \Sigma u,
\qquad
\Sigma = \frac{1}{n-1}\sum_i x_i x_i^\top = \frac{1}{n-1} X_c^\top X_c
$$

$\Sigma$ is the **covariance matrix** (statistics, chapter 01): symmetric, positive semi-definite (section 01), variance along $u$ is $u^\top\Sigma u$ (section 01's quadratic form).

## Solution: An Eigenvalue Problem in Disguise

Constrained optimization (section 05, chapter 06) — Lagrangian with $\mathcal{L} = u^\top \Sigma u - \lambda(u^\top u - 1)$:

$$
\nabla_u \mathcal{L} = 2\Sigma u - 2\lambda u = \mathbf{0}
\;\;\Longrightarrow\;\;
\boxed{\;\Sigma u = \lambda u\;}
$$

**The optimal direction is an eigenvector of $\Sigma$.** Substitute back into the objective:

$$
u^\top \Sigma u = u^\top (\lambda u) = \lambda\,(u^\top u) = \lambda
$$

— the variance along eigenvector $u$ **equals its eigenvalue**. Maximizing variance = picking the **largest eigenvalue**; the corresponding eigenvector is the **first principal component**. Take eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0$ (PSD ⇒ non-negative, section 01): first $k$ eigenvectors $U_k = [u_1, \ldots, u_k]$ form the new coordinates.

- **Projection:** $z = U_k^\top x \in \mathbb{R}^k$ — the compressed representation.
- **Reconstruction:** $\hat x = U_k z$ (back to $d$ dimensions, in the $k$-dimensional subspace).
- **Explained variance ratio:** $\lambda_j / \sum_{i} \lambda_i$ — the fraction of total variance kept by component $j$ (total variance $= \operatorname{tr}\Sigma = \sum \lambda_i$, section 01 — the trace-invariance fact, now with meaning).

## Worked Example (2D)

Covariance matrix:

$$
\Sigma = \begin{bmatrix} 2 & 1.6 \\ 1.6 & 2 \end{bmatrix}
$$

Eigenstructure (section 01's pattern for $\begin{bmatrix} a & b \\ b & a \end{bmatrix}$: eigenvectors $(1,1)$, $(1,-1)$):

$$
\Sigma \begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}3.6\\3.6\end{bmatrix} = 3.6\begin{bmatrix}1\\1\end{bmatrix},
\qquad
\Sigma \begin{bmatrix}1\\-1\end{bmatrix} = \begin{bmatrix}0.4\\-0.4\end{bmatrix} = 0.4\begin{bmatrix}1\\-1\end{bmatrix}
$$

$\lambda_1 = 3.6$ with $u_1 = \tfrac{1}{\sqrt2}(1, 1)$; $\lambda_2 = 0.4$ with $u_2 = \tfrac{1}{\sqrt2}(1, -1)$.

- PC1: data's main spread is along the diagonal $x_1 \approx x_2$ (the strong positive correlation: off-diagonal $1.6$).
- **Keep one component:** variance captured $3.6$, ratio $3.6/(3.6 + 0.4) = 3.6/4 = \mathbf{90\%}$; the orthogonal direction holds $10\%$ — dropped, reconstruction error exactly that residual variance.
- Compression: $2 \to 1$ dimension at $90\%$ variance retention.

```python
# eigen of [[a,b],[b,a]] is (a+b, a-b) with vectors (1,1),(1,-1)
a, b = 2, 1.6
lam1, lam2 = a + b, a - b
total = a + a
print(lam1, lam2, lam1/total)    # 3.6 0.4 0.9
```

## Properties and Connections

1. **Components are uncorrelated:** scores $z_j = u_j^\top x$ have zero covariance for $j \ne l$ — because $\Sigma$ is symmetric with *orthogonal* eigenvectors (section 01: the spectral theorem), and $\operatorname{Cov}(z_j, z_l) = u_j^\top \Sigma u_l = \lambda_l\, u_j^\top u_l = 0$. PCA decorrelates.
2. **Order by variance:** $\lambda_1 \ge \lambda_2 \ge \cdots$ — component 1 carries the most information in the variance sense; dropping the tail is optimal in *mean squared reconstruction error* (Eckart–Young: best rank-$k$ approximation — the SVD theorem from section 01; the optimal low-rank view of the data matrix is PCA's reconstruction).
3. **SVD link (how it is computed):** $X_c = USV^\top$ ⇒ $\Sigma \propto X_c^\top X_c = V S^2 V^\top/(n-1)$: the right singular vectors $V$ are the principal directions, $S^2/(n-1)$ the eigenvalues. Numerically, software runs SVD, never explicitly inverts anything — section 01's practical punchline.
4. **Scale dependence:** variance is measured in the units of each feature — a feature with large units dominates PC1. **Standardize first** ($z$-scores, section 04) unless the raw scale is meaningful.
5. **What PCA is not:** unsupervised (never sees labels — it preserves variance, which may be variance about *irrelevant* structure), nonlinear (curved manifolds need kernels/autoencoders), and a guarantee of interpretability (components are mixtures of original features).

## Connection to AI

- **Dimensionality reduction** for visualization (project to 2D), noise filtering (drop low-variance components — assumed noise), and preprocessing before downstream models (smaller, decorrelated inputs).
- **Embeddings:** word/document/image embeddings are routinely analyzed with PCA (e.g., offset directions like "king − queen ≈ gender" are linear structure PCA-family tools expose); embeddings are high-dimensional and correlated — PCA's vocabulary ("variance explained") is standard in representation analysis.
- **Whitening** (scale so $\Sigma = I$ using eigenvalues) stabilizes downstream training — same eigenvectors, rescaled.
- **The spectral theorem in production:** any time a symmetric matrix's eigenvectors define your algorithm (PCA, spectral clustering, attention analysis), section 01 is the load-bearing wall.
- **Beyond linear:** kernel PCA, autoencoders (section 08) learn nonlinear analogs — but "compress by keeping the highest-variance structure" remains the goal statement.

## Common Mistakes

- Forgetting to center (and to standardize when scales differ) — PCA on uncentered data finds the mean direction (the first component just points at the data's offset).
- Reading "90% variance" as "90% of predictive power for the label" — variance ≠ relevance to the task; supervised methods (LDA, etc.) optimize different criteria.
- Interpreting component signs/rotations as meaningful — eigenvectors are defined up to sign, and equal eigenvalues allow arbitrary rotation within their subspace.
- Choosing $k$ by rule of thumb without looking at explained variance — the scree plot / cumulative ratio is the honest tool.
- Applying PCA per-sample instead of per-feature (mixing up rows and columns of $X_c$).
- Expecting orthogonality of *original features* afterward — PCA makes *scores* uncorrelated; it does not change the input features themselves.

## Exercises

1. Set up the maximization and Lagrangian for PCA (state variables and constraint), and derive $\Sigma u = \lambda u$ in three lines.
2. For $\Sigma = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$: eigenvalues, eigenvectors, and the variance ratio of PC1?
3. The chapter's $\Sigma$: verify that $u_1 = \tfrac{1}{\sqrt2}(1,1)$ gives variance $u_1^\top \Sigma u_1 = 3.6$ by direct multiplication.
4. Why are PC scores uncorrelated? (One sentence: eigenvectors of a symmetric matrix + the covariance formula.)
5. Data standardized ($\Sigma$ has diagonal entries $1$): what is $\operatorname{tr}(\Sigma)$ for $d = 5$, and what does $\sum\lambda_i$ equal?
6. Explain: dropping PC2 in the worked example costs exactly what in reconstruction MSE? (Reference $10\%$.)
