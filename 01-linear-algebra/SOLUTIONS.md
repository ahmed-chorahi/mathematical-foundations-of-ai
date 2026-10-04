# 01 — Linear Algebra: Solutions

## 01-vectors.md

**1. $\mathbf{u} + \mathbf{v}$ and $2\mathbf{u} - 3\mathbf{v}$ for $\mathbf{u} = (1,2,3)$, $\mathbf{v} = (4,-1,0)$.**
$$
\mathbf{u}+\mathbf{v} = (1+4,\; 2-1,\; 3+0) = (5, 1, 3)
$$
$$
2\mathbf{u} = (2,4,6),\quad 3\mathbf{v} = (12,-3,0),\quad
2\mathbf{u}-3\mathbf{v} = (2-12,\; 4+3,\; 6-0) = (-10, 7, 6)
$$

**2. Standard basis of $\mathbb{R}^3$; one-hot for class 2 of 4.**
$$
\mathbf{e}_1 = (1,0,0),\quad \mathbf{e}_2 = (0,1,0),\quad \mathbf{e}_3 = (0,0,1)
$$
One-hot, class 2 of 4 (1-indexed): $(0, 1, 0, 0)$.

**3. Express $(7,-1)$ in terms of $\mathbf{e}_1, \mathbf{e}_2$.**
$(7,-1) = 7(1,0) + (-1)(0,1) = 7\mathbf{e}_1 - \mathbf{e}_2$.
The coefficients are just the components — the standard basis makes coordinates read off directly.

**4. Commutativity of vector addition.**
$(\mathbf{x}+\mathbf{y})_i = x_i + y_i = y_i + x_i = (\mathbf{y}+\mathbf{x})_i$ for every index $i$; equal components means equal vectors.

**5. $\mathbf{z} = 2\mathbf{x} + \mathbf{b}$ with $\mathbf{x} = (1,3)$, $\mathbf{b} = (-1,5)$.**
$2\mathbf{x} = (2,6)$, so $\mathbf{z} = (2-1,\; 6+5) = (1, 11)$.

**6. The set $\{c\mathbf{v} : c \in \mathbb{R}\}$ for $\mathbf{v} \ne \mathbf{0}$.**
A straight line through the origin in direction $\mathbf{v}$ — both directions, since $c$ ranges over negative values too. (If restricted to $c \ge 0$: a ray.) This is $\operatorname{span}(\mathbf{v})$.

## 02-dot-product-and-norms.md

**1. $\mathbf{x}\cdot\mathbf{y}$ for $(2,-1,5)$, $(3,4,-2)$.**
$2\cdot3 + (-1)\cdot4 + 5\cdot(-2) = 6 - 4 - 10 = -8$.

**2. $\|\mathbf{x}\|_2$, $\|\mathbf{x}\|_1$ for $(3,-4,0)$.**
$$
\|\mathbf{x}\|_2 = \sqrt{9 + 16 + 0} = \sqrt{25} = 5,
\qquad
\|\mathbf{x}\|_1 = 3 + 4 + 0 = 7
$$

**3. Are $(1,2)$ and $(2,-1)$ orthogonal?**
$\mathbf{u}\cdot\mathbf{v} = 1\cdot2 + 2\cdot(-1) = 2 - 2 = 0$. Both non-zero and the dot product is zero → orthogonal ✓ (they are perpendicular — sketch to see it).

**4. Distance between $(1,1)$ and $(4,5)$.**
$d = \sqrt{(4-1)^2 + (5-1)^2} = \sqrt{9+16} = 5$.

**5. Cosine similarity of $(1,0)$ and $(1,1)$; angle.**
$\|\mathbf{x}\| = 1$, $\|\mathbf{y}\| = \sqrt{2}$, $\mathbf{x}\cdot\mathbf{y} = 1$:
$$
\cos\theta = \frac{1}{1\cdot\sqrt2} = \frac{\sqrt2}{2} \approx 0.7071
\quad\Rightarrow\quad \theta = 45° \;\left(\frac{\pi}{4}\right)
$$

**6. $\mathbf{0}\cdot\mathbf{y} = \mathbf{0}$, and the geometric reading.**
$\mathbf{0}\cdot\mathbf{y} = \sum_i 0\cdot y_i = 0$. Consistency with $\mathbf{x}\cdot\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$: the right side is $\|\mathbf{0}\|\|\mathbf{y}\|\cos\theta = 0$ for *any* $\theta$ — both sides are zero, and the identity imposes no constraint on the angle (the zero vector has no direction).

**7. `cosine(x, y)` in pure Python.**
```python
def cosine(x, y):
    num = sum(a * b for a, b in zip(x, y))
    den = (sum(v * v for v in x) ** 0.5) * (sum(v * v for v in y) ** 0.5)
    return num / den

print(cosine([1, 0], [1, 1]))   # 0.7071067811865475
```

## 03-matrices.md

**1. $\begin{bmatrix}2&0\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix}$.**
Row 1: $2\cdot1 + 0\cdot2 = 2$. Row 2: $1\cdot1 + 3\cdot2 = 7$. Result: $\begin{bmatrix}2\\7\end{bmatrix}$.

**2. $AB$ vs $BA$.**
$$
AB = \begin{bmatrix}1&0\\0&2\end{bmatrix}\begin{bmatrix}3&1\\0&1\end{bmatrix} = \begin{bmatrix}3&1\\0&2\end{bmatrix}, \qquad
BA = \begin{bmatrix}3&1\\0&1\end{bmatrix}\begin{bmatrix}1&0\\0&2\end{bmatrix} = \begin{bmatrix}3&2\\0&2\end{bmatrix}
$$
$AB \ne BA$ — multiplication is not commutative.

**3. Transposes.**
$\begin{bmatrix}1&-1\\0&5\end{bmatrix}^\top = \begin{bmatrix}1&0\\-1&5\end{bmatrix}$ (row 1 becomes column 1).
$(1,2)^\top = \begin{bmatrix}1\\2\end{bmatrix}$ is already a column; its transpose is the row $\begin{bmatrix}1&2\end{bmatrix}$.

**4. Inverse of $\begin{bmatrix}2&5\\1&3\end{bmatrix}$.**
$\det = 2\cdot3 - 5\cdot1 = 1$, so
$$
A^{-1} = \frac{1}{1}\begin{bmatrix}3&-5\\-1&2\end{bmatrix} = \begin{bmatrix}3&-5\\-1&2\end{bmatrix}
$$
Verify:
$$
\begin{bmatrix}2&5\\1&3\end{bmatrix}\begin{bmatrix}3&-5\\-1&2\end{bmatrix}
= \begin{bmatrix}6-5 & -10+10\\3-3 & -5+6\end{bmatrix} = \begin{bmatrix}1&0\\0&1\end{bmatrix} \;✓
$$

**5. $\begin{bmatrix}1&2\\2&4\end{bmatrix}$ singular?**
$\det = 1\cdot4 - 2\cdot2 = 0$ → singular, no inverse.
What goes wrong: the second column equals $2\times$ the first, so $A\mathbf{x} = 2x_1\mathbf{c}_1 + x_2(2\mathbf{c}_1)$ — only multiples of $(1,2)$ are reachable, one dimension is lost, and inputs differing by the dependent direction give the same output.

**6. Rotation of $(1,0)$.**
$\begin{bmatrix}0&-1\\1&0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix}$: a $90°$ counterclockwise rotation ✓.

**7. $(AB)^\top = B^\top A^\top$ for the chapter's $A, B$.**
From the chapter, $AB = \begin{bmatrix}2&1\\4&3\end{bmatrix}$, so $(AB)^\top = \begin{bmatrix}2&4\\1&3\end{bmatrix}$.
$B^\top = \begin{bmatrix}0&1\\1&0\end{bmatrix}$ (B is symmetric), $A^\top = \begin{bmatrix}1&3\\2&4\end{bmatrix}$:
$$
B^\top A^\top = \begin{bmatrix}0&1\\1&0\end{bmatrix}\begin{bmatrix}1&3\\2&4\end{bmatrix}
= \begin{bmatrix}0+2 & 0+4\\1+0 & 3+0\end{bmatrix} = \begin{bmatrix}2&4\\1&3\end{bmatrix} \;✓
$$

## 04-systems-rank-and-basis.md

**1. $2x_1 + x_2 = 7$, $x_1 - x_2 = 2$.**
Add the equations: $3x_1 = 9 \Rightarrow x_1 = 3$. Then $x_2 = x_1 - 2 = 1$.
Check: $2(3) + 1 = 7$ ✓, $3 - 1 = 2$ ✓.

**2. $A\mathbf{x} = (3,6)$ with $A = \begin{bmatrix}1&2\\2&4\end{bmatrix}$.**
Second equation is $2\times$ the first ($2x_1 + 4x_2 = 6 = 2\cdot3$) — both say $x_1 + 2x_2 = 3$, so $x_1 = 3 - 2x_2$ with $x_2$ free.
Solutions: $(3, 0)$ (set $x_2 = 0$) and $(1, 1)$ (set $x_2 = 1$); in general the whole line $\{(3-2t, t) : t \in \mathbb{R}\}$.

**3. Independence of $(1,1,0)$, $(0,2,1)$.**
Solve $c_1(1,1,0) + c_2(0,2,1) = (0,0,0)$: the third coordinate gives $c_2 = 0$; the first gives $c_1 = 0$. Only the trivial combination → **independent**.

**4. Spans.**
$\operatorname{span}((1,0),(1,0))$ = all $c(1,0)$ = the x-axis (a line) — the duplicate adds nothing.
$\operatorname{span}((1,0),(0,1))$ = $\{(c_1, c_2)\}$ = all of $\mathbb{R}^2$.

**5. Is $\{(1,0,0),(0,1,0)\}$ a basis of $\mathbb{R}^3$?**
Independent: yes (neither is a multiple of the other; the usual check gives $c_1 = c_2 = 0$).
Spanning: no — no combination reaches $(0,0,1)$ (the third coordinate is always 0).
Fails condition 2 → **not a basis of $\mathbb{R}^3$** (it is a basis of the $xy$-plane subspace).

**6. Rank of $\begin{bmatrix}1&3&0\\2&6&1\\0&0&2\end{bmatrix}$.**
$R_2 \leftarrow R_2 - 2R_1$: $\begin{bmatrix}1&3&0\\0&0&1\\0&0&2\end{bmatrix}$. Then $R_3 \leftarrow R_3 - 2R_2$: $\begin{bmatrix}1&3&0\\0&0&1\\0&0&0\end{bmatrix}$.
Pivots in columns 1 and 3 → **rank = 2**.

**7. $H = \{\mathbf{x} : x_1+x_2+x_3 = 0\}$ is a subspace; $\dim H = 2$.**
- $\mathbf{0} \in H$: $0+0+0 = 0$ ✓
- Closed under addition: if $x_1+x_2+x_3 = 0$ and $y_1+y_2+y_3 = 0$, their sum has coordinate sum $0 + 0 = 0$ ✓
- Closed under scaling: coordinate sum of $c\mathbf{x}$ is $c\cdot0 = 0$ ✓

Independence of $(1,-1,0)$, $(0,1,-1)$: $c_1(1,-1,0) + c_2(0,1,-1) = (c_1, -c_1+c_2, -c_2) = \mathbf{0}$ forces $c_2 = 0$ (third coordinate), then $c_1 = 0$ (first) ✓

Spanning: any solution has $x_3 = -x_1 - x_2$, so
$$
(x_1, x_2, -x_1-x_2) = x_1(1,-1,0) + (x_1+x_2)(0,1,-1)
$$
(check: first coordinate $x_1$ ✓; second $-x_1 + x_1 + x_2 = x_2$ ✓; third $-(x_1+x_2)$ ✓).
Independent spanning set of size 2 → basis, so $\dim H = 2$ (one equation removes one dimension from $\mathbb{R}^3$).

## 05-orthogonality-and-least-squares.md

**1. Project $(4,2)$ onto $(1,0)$.**
$c = \frac{4\cdot1 + 2\cdot0}{1} = 4$, so $\mathbf{p} = (4,0)$. Residual $(4,2)-(4,0) = (0,2)$; $(0,2)\cdot(1,0) = 0$ ✓ orthogonal.

**2. Project $(2,3)$ onto span$(1,1)$; distance.**
$c = \frac{2+3}{2} = \frac{5}{2}$, $\mathbf{p} = (2.5, 2.5)$.
$$
d = \|(2,3)-(2.5,2.5)\| = \sqrt{0.25+0.25} = \sqrt{0.5} \approx 0.707
$$

**3. Projection of a vector already on the line.**
If $\mathbf{x} = c\mathbf{u}$, then $\operatorname{proj}_\mathbf{u}\mathbf{x} = \frac{c\mathbf{u}\cdot\mathbf{u}}{\mathbf{u}\cdot\mathbf{u}}\mathbf{u} = \frac{c(\mathbf{u}\cdot\mathbf{u})}{\mathbf{u}\cdot\mathbf{u}}\mathbf{u} = c\mathbf{u} = \mathbf{x}$ ✓ (projection is idempotent — it changes nothing when already in the subspace).

**4. Fit $y = b + wx$ to $(1,3)$, $(2,5)$.**
$$
X^\top X = \begin{bmatrix}2&3\\3&5\end{bmatrix}, \quad
X^\top\mathbf{y} = \begin{bmatrix}8\\13\end{bmatrix}, \quad
\det = 10 - 9 = 1
$$
$$
(X^\top X)^{-1} = \begin{bmatrix}5&-3\\-3&2\end{bmatrix}, \qquad
\mathbf{w} = \begin{bmatrix}5&-3\\-3&2\end{bmatrix}\begin{bmatrix}8\\13\end{bmatrix}
= \begin{bmatrix}40-39\\-24+26\end{bmatrix} = \begin{bmatrix}1\\2\end{bmatrix}
$$
Line: $\hat y = 1 + 2x$. Check: $1+2 = 3$ ✓ at $x=1$; $1+4 = 5$ ✓ at $x=2$. Residuals are $0,0$ — two points determine a line exactly.

**5. $A = \begin{bmatrix}1\\1\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}1\\3\end{bmatrix}$.**
$A^\top A = 1\cdot1 + 1\cdot1 = 2$. $A^\top\mathbf{b} = 1\cdot1 + 1\cdot3 = 4$. Solution: $w = 4/2 = 2$.
$\hat{\mathbf{y}} = (2,2)$. Residual: $\mathbf{b} - \hat{\mathbf{y}} = (-1, 1)$.
Orthogonality: $(-1)(1) + (1)(1) = 0$ ✓ — the defining condition of least squares.

**6. Best through-origin line vs. fitted line.**
$w = \frac{1\cdot2 + 2\cdot3 + 3\cdot5}{1 + 4 + 9} = \frac{23}{14}$. Residuals:
$$
2 - \tfrac{23}{14} = \tfrac{5}{14}, \quad
3 - \tfrac{46}{14} = -\tfrac{4}{14}, \quad
5 - \tfrac{69}{14} = \tfrac{1}{14}
$$
$$
\text{SSE} = \frac{25 + 16 + 1}{196} = \frac{42}{196} = \frac{3}{14} \approx 0.214
$$
Compare: $3/14 \approx 0.214 > 1/6 \approx 0.167$ from the line with intercept. Concluding: a constrained model (forced through the origin) cannot beat a more flexible one that *contains* it — the extra freedom of the intercept can only reduce the minimum error (and here it does).

## 06-eigenvalues.md

**1. Eigenvalue of $(1,-1)$ for $\begin{bmatrix}2&1\\1&2\end{bmatrix}$.**
$$
A\mathbf{v} = \begin{bmatrix}2(1)+1(-1)\\1(1)+2(-1)\end{bmatrix} = \begin{bmatrix}1\\-1\end{bmatrix} = 1\cdot\mathbf{v}
\quad\Rightarrow\quad \lambda = 1
$$

**2. Eigenvalues of $\begin{bmatrix}3&0\\0&5\end{bmatrix}$.**
$\lambda = 3$ and $\lambda = 5$: for a diagonal matrix, $D\mathbf{e}_i = d_{ii}\mathbf{e}_i$ — each standard basis vector is already an eigenvector with the diagonal entry as eigenvalue. No polynomial needed.

**3. $\begin{bmatrix}4&1\\0&3\end{bmatrix}$.**
Upper triangular → eigenvalues are the diagonal entries $4$ and $3$ (since $\det(A-\lambda I) = (4-\lambda)(3-\lambda)$).
- $\lambda = 4$: $(A-4I)\mathbf{v} = \begin{bmatrix}0&1\\0&0\end{bmatrix}\mathbf{v} = \mathbf{0} \Rightarrow v_2 = 0 \Rightarrow \mathbf{v} = (1,0)$.
Check: $A(1,0) = (4,0)$ ✓
- $\lambda = 3$: $(A-3I)\mathbf{v} = \begin{bmatrix}1&1\\0&0\end{bmatrix}\mathbf{v} = \mathbf{0} \Rightarrow v_2 = -v_1 \Rightarrow \mathbf{v} = (1,-1)$.
Check: $A(1,-1) = (4-1, 0-3) = (3,-3) = 3(1,-1)$ ✓
Not symmetric: $a_{12} = 1 \ne 0 = a_{21}$.

**4. Positive definiteness of $\begin{bmatrix}2&1\\1&2\end{bmatrix}$.**
Eigenvalues $1$ and $3$, both $> 0$ → **positive definite** ✓ (consistent with $\mathbf{x}^\top A\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 = x_1^2 + x_2^2 + (x_1+x_2)^2 \ge 0$, which is zero only when $x_1 = x_2 = 0$).

**5. $A\mathbf{v} = \lambda\mathbf{v} \Rightarrow A^2\mathbf{v} = \lambda^2\mathbf{v}$.**
$$
A^2\mathbf{v} = A(A\mathbf{v}) = A(\lambda\mathbf{v}) = \lambda A\mathbf{v} = \lambda(\lambda\mathbf{v}) = \lambda^2\mathbf{v}
$$
By induction: $A^k\mathbf{v} = \lambda^k\mathbf{v}$ — the eigenvector stays fixed, the eigenvalue carries the whole power.

**6. Symmetric with eigenvalues $-2$ and $4$.**
Not positive definite: the eigenvalue test requires **all** eigenvalues positive, and $-2 < 0$. For the eigenvector $\mathbf{v}$ of $-2$: $\mathbf{v}^\top A\mathbf{v} = -2\|\mathbf{v}\|^2 < 0$ — the quadratic form is negative in that direction.

**7. $B = \begin{bmatrix}1&0\\0&-1\end{bmatrix}$, $\mathbf{x} = (0,1)$.**
$B\mathbf{x} = (0,-1)$, so $\mathbf{x}^\top B\mathbf{x} = 0 + (1)(-1) = -1 < 0$ → **not positive definite** (indefinite: it is $+1$ along $(1,0)$ and $-1$ along $(0,1)$).

## 07-svd.md

**1. Singular values of $\begin{bmatrix}5\\0\end{bmatrix}$.**
$A^\top A = \begin{bmatrix}5&0\end{bmatrix}\begin{bmatrix}5\\0\end{bmatrix} = [25]$. Eigenvalue $25$ → $\sigma = \sqrt{25} = 5$.
(One singular value — a matrix with a single column has at most one.)

**2. When does truncation pay off?**
$m = n = 1000$, $k = 10$: factors cost $k(m+n+1) = 10 \cdot 2001 = 20{,}010$ numbers vs. $mn = 1{,}000{,}000$ — about **50× smaller**. Conclusion: storage savings require $k \ll \min(m,n)$ on a matrix large enough that the per-factor overhead ($m + n + 1$) is small relative to $mn$. The tiny example in the chapter saves nothing; real data matrices do.

**3. SVD of $\begin{bmatrix}0&0\\3&0\end{bmatrix}$.**
$A^\top A = \begin{bmatrix}0&3\\0&0\end{bmatrix}\begin{bmatrix}0&0\\3&0\end{bmatrix} = \begin{bmatrix}9&0\\0&0\end{bmatrix}$ → eigenvalues $9, 0$ → $\sigma_1 = 3$, $\sigma_2 = 0$; $V = I$ (eigenvectors $\mathbf{e}_1, \mathbf{e}_2$ in order).
$\mathbf{u}_1 = A\mathbf{v}_1/\sigma_1 = (0,3)/3 = (0,1)$.
$\sigma_2 = 0$: $A\mathbf{v}_2 = \mathbf{0}$, so $\mathbf{u}_2$ is any unit vector orthogonal to $(0,1)$ — choose $(1,0)$.
$$
U = \begin{bmatrix}0&1\\1&0\end{bmatrix}, \quad
\Sigma = \begin{bmatrix}3&0\\0&0\end{bmatrix}, \quad
V = \begin{bmatrix}1&0\\0&1\end{bmatrix}
$$
Verify: $U\Sigma = \begin{bmatrix}0&0\\3&0\end{bmatrix}$, times $V^\top = I$ gives $\begin{bmatrix}0&0\\3&0\end{bmatrix} = A$ ✓
(The arbitrary choice for $\mathbf{u}_2$ shows singular vectors are not unique when $\sigma = 0$.)

**4. $\|A\mathbf{v}_1\| = \sigma_1$ in the chapter example.**
$A\mathbf{v}_1 = \begin{bmatrix}1&0\\0&2\\0&0\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}0\\2\\0\end{bmatrix}$, and $\|\begin{bmatrix}0\\2\\0\end{bmatrix}\| = 2 = \sigma_1$ ✓

**5. Rank from singular values $9, 4, 0, 0, 0, 0, 0$.**
Count the positive ones: $9$ and $4$ → **rank = 2** (of a possible $\min(4,7) = 4$).

**6. "Invertible ⇒ all singular values non-zero."**
**True.** If some $\sigma_i = 0$, then $A\mathbf{v}_i = \sigma_i\mathbf{u}_i = \mathbf{0}$ for a unit $\mathbf{v}_i$ — a non-zero vector in the null space, which means $A$ is singular. Contrapositive: invertible ⇒ no zero singular values. (For square matrices: exactly $n$ positive singular values.)
