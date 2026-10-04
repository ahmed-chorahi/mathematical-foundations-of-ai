# 02 — Calculus: Solutions

## 01-limits-and-continuity.md

**1. $\lim_{x\to3}(x^2+2x)$.**
Both $x^2$ and $2x$ have limits at $3$ (they are continuous): $9 + 6 = 15$. By the sum law the limit of the sum is the sum of the limits; equivalently, substitute into the continuous polynomial.

**2. $\lim_{x\to2}\frac{x^2-4}{x-2}$.**
Factor: $\frac{(x-2)(x+2)}{x-2} = x+2$ for $x \ne 2$; limit $= 2 + 2 = 4$.
(The cancellation is legal *in the limit* because the limit never evaluates at $x = 2$.)

**3. $\lim_{x\to0}\frac{|x|}{x}$.**
$x > 0$: $\frac{x}{x} = 1 \Rightarrow$ right limit $= 1$.
$x < 0$: $\frac{-x}{x} = -1 \Rightarrow$ left limit $= -1$.
Different one-sided limits → **the limit does not exist**.

**4. $f(x) = \frac{1}{x-1}$ at $x = 1$.**
Not continuous — condition 1 fails first: $f(1)$ is undefined (division by zero). Conditions 2 and 3 fail consequently. The break is at a point excluded from the domain; the function is continuous everywhere else.

**5. Is ReLU continuous at $0$?**
- $f(0) = \max(0,0) = 0$ defined ✓
- $\lim_{x\to0^-} 0 = 0$ and $\lim_{x\to0^+} x = 0$ — equal, so the limit is $0$ ✓
- limit $= f(0)$ ✓ → **continuous at $0$** (but not differentiable there — left slope $0$, right slope $1$).

**6. Limit exists where the function is undefined.**
$f(x) = \frac{x^2-1}{x-1}$ at $x = 1$: undefined at $1$ ($0/0$), but $\lim_{x\to1} f(x) = \lim (x+1) = 2$.

## 02-derivatives.md

**1. $f(x) = 4x^3 - 2x + 7$.**
$f'(x) = 12x^2 - 2$ (power rule term by term; the constant differentiates to $0$).

**2. $f'(2)$ for $f(x) = x^2$ from the definition.**
$$
\frac{(2+h)^2 - 2^2}{h} = \frac{4 + 4h + h^2 - 4}{h} = 4 + h \;\xrightarrow{h\to0}\; 4
$$

**3. $g(x) = (2x+1)^5$.**
Outer power, inner $u = 2x+1$: $g'(x) = 5(2x+1)^4 \cdot 2 = 10(2x+1)^4$.

**4. $h(x) = xe^x$.**
Product rule: $h'(x) = 1\cdot e^x + x\cdot e^x = (1+x)e^x$.

**5. $\frac{d}{dx}\frac{1}{x} = -\frac{1}{x^2}$.**
From the definition: $\frac{\frac{1}{x+h} - \frac{1}{x}}{h} = \frac{x - (x+h)}{hx(x+h)} = \frac{-1}{x(x+h)} \to -\frac{1}{x^2}$ as $h \to 0$.
(Or the power rule: $x^{-1} \to -x^{-2}$.)

**6. $\sigma'(0)$ and $\sigma'(5)$.**
$\sigma'(x) = \sigma(x)(1-\sigma(x))$; at $x = 0$: $\sigma(0) = 0.5$, so $\sigma'(0) = 0.5 \cdot 0.5 = 0.25$ — a unit change of input passes through at quarter strength (the sigmoid's maximum slope).
At $x = 5$: $\sigma(5) \approx 0.993$, so $\sigma'(5) \approx 0.993 \cdot 0.007 \approx 0.007$ — under $1\%$ passes through. Saturated sigmoid units barely transmit gradients.

**7. $f(x) = \ln(x^2+1)$.**
$f'(x) = \frac{1}{x^2+1}\cdot 2x = \frac{2x}{x^2+1}$; at $x = 0$: $f'(0) = 0$.

## 03-partial-derivatives-and-gradient.md

**1. $f(x,y) = x^2y + y^3$.**
$\dfrac{\partial f}{\partial x} = 2xy$ ($y$ constant), $\dfrac{\partial f}{\partial y} = x^2 + 3y^2$.

**2. Evaluated at $(2,1)$.**
$f_x = 2(2)(1) = 4$, $f_y = 2^2 + 3(1)^2 = 7$.

**3. $\nabla f$ for $f = e^{x+y}$ at $(0,0)$.**
$f_x = e^{x+y}$, $f_y = e^{x+y}$; at $(0,0)$: $\nabla f = (1, 1)^\top$, length $\sqrt{2} \approx 1.414$.

**4. $\nabla f = (4,0)$, rate along $\mathbf{u} = (0,1)$.**
$(4,0)\cdot(0,1) = 0$ — no first-order change: the direction of $\mathbf{u}$ runs along a contour of $f$ (perpendicular to the gradient).

**5. $L(w,b) = (wx+b-y)^2$, $x=1$, $y=3$, at $(w,b) = (2,0)$.**
$\hat y = 2\cdot1 + 0 = 2$, so $\hat y - y = -1$.
$$
\frac{\partial L}{\partial w} = 2(\hat y - y)\cdot x = 2(-1)(1) = -2,
\qquad
\frac{\partial L}{\partial b} = 2(\hat y - y)\cdot 1 = -2
$$
(Structure: loss derivative × prediction derivative — the chain rule in miniature.)

**6. First-order condition for an interior local extremum.**
$f_x = 0$ and $f_y = 0$, i.e., $\nabla f = \mathbf{0}$ — every partial vanishes (for a differentiable $f$ at an interior point).

## 04-jacobian-and-hessian.md

**1. Jacobian of $F(x,y) = (x^2, xy, y)^\top$.**
$$
J_F = \begin{bmatrix} 2x & 0 \\ y & x \\ 0 & 1 \end{bmatrix}, \qquad \text{shape } 3 \times 2
$$
(3 outputs × 2 inputs.)

**2. $f = x^2 + 3xy + y^3$ at $(1,1)$.**
$f_x = 2x + 3y \Rightarrow 5$ at $(1,1)$; $f_y = 3x + 3y^2 \Rightarrow 6$ at $(1,1)$.
$$
J_f = \begin{bmatrix} 5 & 6 \end{bmatrix}, \qquad
\nabla f = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
$$

**3. $J_F$ for $F(u,v) = (u+v, uv)$ at $(2,3)$.**
$$
J_F = \begin{bmatrix} 1 & 1 \\ v & u \end{bmatrix}
= \begin{bmatrix} 1 & 1 \\ 3 & 2 \end{bmatrix}
$$
Entry $J_{21} = \partial(uv)/\partial u = v = 3$: at $(2,3)$, increasing $u$ by a small amount (keeping $v$ fixed) increases the second output $uv$ by about $3$ times that amount.

**4. Hessian at $(0,0)$ and $(0,1)$.**
$f_{xx} = 2$, $f_{xy} = f_{yx} = 3$, $f_{yy} = 6y$, so
$$
H = \begin{bmatrix} 2 & 3 \\ 3 & 6y \end{bmatrix}
\;\Rightarrow\;
H(0,0) = \begin{bmatrix}2&3\\3&0\end{bmatrix},\quad
H(0,1) = \begin{bmatrix}2&3\\3&6\end{bmatrix}
$$
Both symmetric ✓ (mixed entries $3 = 3$ at each point).

**5. Eigenvalues $2, 5, -1$ at a critical point.**
Mixed signs → Hessian indefinite → **saddle point**: the function curves up along the eigenvectors of $2$ and $5$, down along the eigenvector of $-1$; a downhill direction exists.

**6. Mixed partials of $xy^2$.**
$$
f_x = y^2 \Rightarrow \frac{\partial}{\partial y}(y^2) = 2y;
\qquad
f_y = 2xy \Rightarrow \frac{\partial}{\partial x}(2xy) = 2y
$$
Equal ($2y = 2y$) ✓ — consistent with symmetry of the Hessian for smooth functions.

## 05-chain-rule.md

**1. $f(x) = (x^2+1)^3$.**
Outer $u^3$, inner $u = x^2+1$: $f'(x) = 3(x^2+1)^2 \cdot 2x = 6x(x^2+1)^2$.

**2. $g(x) = e^{-3x}$.**
Outer $e^u$, inner $u = -3x$: $g'(x) = e^{-3x}\cdot(-3) = -3e^{-3x}$.

**3. $z = xy$, $x = t^2$, $y = \sin t$.**
$$
\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}
= y\cdot 2t + x\cdot\cos t = 2t\sin t + t^2\cos t
$$
Check via substitution: $z = t^2\sin t$; product rule: $2t\sin t + t^2\cos t$ ✓

**4. $L = (\hat y - y)^2$, $\hat y = w_1x$.**
$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial w_1}
= 2(\hat y - y)\cdot x
$$
Two local factors: loss slope $2(\hat y - y)$, prediction slope $x$.

**5. $a = \sigma(w_1x)$, $L = (a-y)^2$ — three factors.**
$$
\frac{\partial L}{\partial w_1}
= \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w_1}
= 2(a-y)\cdot\sigma'(w_1x)\cdot x
$$
with $z = w_1x$. Loss slope × activation slope × input.

**6. Evaluate at $x=1$, $w_1=0$, $y=0$.**
$a = \sigma(0) = 0.5$; $\sigma'(0) = 0.5\cdot0.5 = 0.25$.
$$
\frac{\partial L}{\partial w_1} = 2(0.5 - 0)\cdot 0.25\cdot 1 = 0.25
$$

## 06-taylor-approximation.md

**1. $f(x) = \sqrt{x}$ at $x = 4$, $h = 0.1$, first order.**
$f(4) = 2$, $f'(x) = \frac{1}{2\sqrt x} \Rightarrow f'(4) = 0.25$.
Approximation: $2 + 0.25(0.1) = 2.025$.
Exact $\sqrt{4.1} \approx 2.02485$; error $\approx 2.025 - 2.02485 = 0.00015$ — small, consistent with $o(h)$ error (specifically, second-order term $-\frac{1}{8}(4)^{-3/2}h^2 = -0.000156$).

**2. Second-order prediction of $f(1.2)$, $f = x^2-4x+5$.**
$f(1) = 2$, $f'(1) = -2$, $f'' = 2$:
$$
2 + (-2)(0.2) + \tfrac12(2)(0.2)^2 = 2 - 0.4 + 0.04 = 1.64
$$
Exact: $1.44 - 4.8 + 5 = 1.64$ ✓ (exact for quadratics).

**3. First-order expansion of $J = \theta_1^2 + 2\theta_2^2$ at $(1,1)$, $\Delta = (0.1, -0.1)$.**
$\nabla J = (2\theta_1, 4\theta_2) = (2,4)$ at $(1,1)$; $J(1,1) = 3$.
$$
J \approx 3 + (2,4)\cdot(0.1,-0.1) = 3 + 0.2 - 0.4 = 2.8
$$
True: $J(1.1, 0.9) = 1.21 + 2(0.81) = 1.21 + 1.62 = 2.83$.
Error $0.03 = \tfrac12\Delta^\top H\Delta$ with $H = \operatorname{diag}(2,4)$: $\tfrac12(2\cdot0.01 + 4\cdot0.01) = 0.03$ ✓

**4. Gradient step on $J = \theta^2$ from $\theta = 3$, $\eta = 0.05$.**
$J'(3) = 6$; $\theta_{\text{new}} = 3 - 0.05(6) = 2.7$; true loss $2.7^2 = 7.29$ (from $9$).
Linear-model prediction: $J(3) + J'(3)(-0.3) = 9 - 1.8 = 7.2$.
Gap $= 7.29 - 7.2 = 0.09 = \tfrac12 J'' (0.3)^2 = \tfrac12(2)(0.09)$ ✓ — exactly the discarded second-order term.

**5. $J(\theta - \eta\nabla J) \approx J(\theta) - \eta\|\nabla J\|^2$.**
First-order expansion with $\Delta = -\eta\nabla J$:
$$
J(\theta + \Delta) \approx J(\theta) + \nabla J^\top\Delta
= J(\theta) + \nabla J^\top(-\eta\nabla J)
= J(\theta) - \eta\|\nabla J\|^2
$$
Since $\|\nabla J\|^2 \ge 0$ and $\eta > 0$, the predicted change is $\le 0$ — strictly negative when $\nabla J \ne \mathbf{0}$.

**6. Why constrain the step?**
$J(\theta) + \nabla J^\top\Delta$ is linear in $\Delta$ with nonzero gradient: without a constraint, $\Delta$ can be sent infinitely far along $-\nabla J/\|\nabla J\|$ and the model value $\to -\infty$ — a meaningless "solution". The bound $\|\Delta\| \le \eta$ (or an equivalent penalty) makes the problem well-posed and reflects reality: the linear model is only trustworthy locally.

## 07-optimization-with-calculus.md

**1. $f(x) = x^2 - 6x + 11$.**
$f'(x) = 2x - 6 = 0 \Rightarrow x = 3$. $f''(3) = 2 > 0$ → **local minimum**; $f(3) = 9 - 18 + 11 = 2$.
(Global too: $f(x) = (x-3)^2 + 2 \ge 2$.)

**2. $f(x) = -x^2 + 4$.**
$f'(x) = -2x = 0 \Rightarrow x = 0$; $f''(0) = -2 < 0$ → **local maximum**, value $4$ (and global — a downward parabola).

**3. $g(x) = x^3 - 3x$.**
$g'(x) = 3x^2 - 3 = 0 \Rightarrow x = \pm1$; $g''(x) = 6x$.
- $x = 1$: $g'' = 6 > 0$ → minimum, $g(1) = 1 - 3 = -2$.
- $x = -1$: $g'' = -6 < 0$ → maximum, $g(-1) = -1 + 3 = 2$.

**4. $f(x,y) = x^2 + y^2$.**
$\nabla f = (2x, 2y) = \mathbf{0}$ only at $(0,0)$. $H = \begin{bmatrix}2&0\\0&2\end{bmatrix}$, eigenvalues $2, 2 > 0$ → positive definite → **local minimum**, $f = 0$ — and global, since $x^2 + y^2 \ge 0$ everywhere.

**5. $f(x,y) = x^2 - y^2$.**
$\nabla f = (2x, -2y) = \mathbf{0}$ at $(0,0)$ ✓. $H = \begin{bmatrix}2&0\\0&-2\end{bmatrix}$, eigenvalues $2$ and $-2$ — mixed signs → **saddle point**.

**6. $f = (x-1)^2 + (y-2)^2$.**
$\nabla f = \begin{pmatrix} 2(x-1) \\ 2(y-2) \end{pmatrix} = \mathbf{0} \Rightarrow (x,y) = (1,2)$.
$f(1,2) = 0$, and $f \ge 0$ for all $(x,y)$ (sum of squares) — the critical value attains the lower bound, so it is **global**.
