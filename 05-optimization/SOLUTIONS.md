# 05 — Optimization: Solutions

## 01-objective-and-loss.md

**1. Logistic-regression objective.** With $p_i = \sigma(w^\top x_i + b)$:

$$
J(w, b) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]
$$

**2. $\partial L/\partial w$ at $x = 2$, $y = 5$, $w = 1$.** $\hat y = wx = 2$; $L = (\hat y - y)^2$, so

$$
\frac{\partial L}{\partial w} = 2(\hat y - y)\,x = 2(2 - 5)(2) = -12
$$

**3. Why SGD can average minibatches.** $J$ is the arithmetic mean of per-example losses, so the minibatch mean is an unbiased estimate of $J$'s gradient ($\mathbb{E}[\text{batch gradient}] = \nabla J$ — chapter 04).

**4. "Maximize validation accuracy" as an objective.** $\max_\theta \text{Acc}(\theta)$ — problematic because accuracy is piecewise constant in $\theta$ (changes only when an argmax flips): gradient $= 0$ almost everywhere, undefined at flips — no direction to descend; training uses a differentiable surrogate (cross-entropy).

**5. Loss when true-class prob is 0.01.** $L = -\ln 0.01 \approx 4.605$ nats ($\approx 6.64$ bits) — a large penalty, which is why confident-but-wrong predictions are heavily penalized by cross-entropy.

**6. Local vs global minimum.** *Local:* smaller than all points in some neighborhood of $\theta^*$. *Global:* smaller than or equal to all points in the domain.

## 02-gradient-descent.md

**1. Two steps on $J(\theta) = \theta^2$, $\theta_0 = 1$, $\eta = 0.5$.** $J' = 2\theta$:
$\theta_1 = 1 - 0.5(2) = 0$; loss $= 0$. $\theta_2 = 0 - 0.5(0) = 0$; loss $= 0$ — one step suffices ($\eta = 1/\lambda = 0.5$ is the exact step for curvature $\lambda = 2$, factor $1 - \eta\lambda = 0$).

**2. $\lambda = 4$: convergence range and $\eta = 0.5$.** Factor $1 - \eta\lambda$ needs $|1 - 4\eta| < 1$, i.e. $0 < \eta < 0.5$. At $\eta = 0.5$: factor $= -1$ — $|\cdot| = 1$: undamped oscillation (magnitude never shrinks or grows); $\eta > 0.5$ diverges.

**3. One step on $J(w) = (2w - 6)^2$ from $w = 0$, $\eta = 0.05$.** $J'(w) = 4(2w - 6)$; at $w = 0$: $-24$; $w_1 = 0 - 0.05(-24) = 1.2$ (optimum is $w = 3$).

**4. Symbolic gradient and one update.** $\partial J/\partial w = 2(wx + b - y)x$, $\partial J/\partial b = 2(wx + b - y)$. At the given point: prediction $= 1\cdot2 + 1 = 3$, error $= 3 - 10 = -7$:
$\partial J/\partial w = 2(-7)(2) = -28$, $\partial J/\partial b = -14$.
Update ($\eta = 0.01$): $w = 1 + 0.28 = 1.28$, $b = 1 + 0.14 = 1.14$.

**5. Why the safe rate depends on curvature.** The quadratic update multiplies $\theta$ by $1 - \eta\lambda$: large $\lambda$ (sharp curvature) makes $|1 - \eta\lambda| \ge 1$ — steps overshoot and amplify; hence $\eta < 2/\lambda_{\max}$, and flat directions tolerate larger steps than sharp ones.

**6. Stop when $|J'| < 10^{-6}$.** Needs $|\theta - 4| < 5\times10^{-7}$; with $\theta_k = 4(1 - 0.8^k)$ that is $0.8^k < 1.25\times10^{-7}$, i.e. $k \ge 72$ — the run stops after **72 iterations** (verified by execution).

## 03-convexity.md

**1. Classify via second derivatives.** $f(x) = x^2$: $f'' = 2 > 0$ — **convex**. $g(x) = -\ln x$: $g'' = 1/x^2 > 0$ on $x>0$ — **convex** (note: $+\ln x$ would be concave). $h(x) = x^3$: $h'' = 6x < 0$ for $x < 0$, $>0$ for $x>0$ — **neither** on $\mathbb{R}$.

**2. $e^x$ convex.** $f''(x) = e^x > 0$ everywhere ⇒ strictly convex on $\mathbb{R}$.

**3. Convexity inequality for $x^2$ at $0, 2$, $t = 1/2$.** LHS: $f(0.5\cdot0 + 0.5\cdot2) = f(1) = 1$. RHS: $0.5 f(0) + 0.5 f(2) = 0 + 0.5(4) = 2$. $1 \le 2$ ✓.

**4. Hessian of $J(w) = \|Xw - y\|^2$.** $J = w^\top X^\top X w - 2y^\top X w + y^\top y$; $\nabla J = 2X^\top(Xw - y)$; $H = 2X^\top X$. For any step $v$: $v^\top H v = 2\|Xv\|^2 \ge 0$ ⇒ PSD for every $X$ ⇒ $J$ convex.

**5. $f(x, y) = xy$ convex?** $H = \begin{bmatrix}0&1\\1&0\end{bmatrix}$; eigenvalues $\pm1$ (or take $v = (1,-1)$: $v^\top Hv = -2 < 0$) — **not PSD** ⇒ not convex (it is a saddle).

**6. Convex vs deep-network guarantees.** Convex training: any converged point is the global optimum, uniquely determined if strictly convex, with convergence theorems. Deep networks: training gives *a* critical point with no global-optimality guarantee — correctness comes from empirical evaluation and theory in progress, not from the objective's shape.

## 04-stochastic-gradient-descent.md

**1. Unbiasedness check ($n = 2$, $L_i = (w - y_i)^2$, $y = (0, 4)$).** $\nabla L_1 = 2w$, $\nabla L_2 = 2(w-4)$; $\nabla J = \tfrac12[2w + 2(w-4)] = 2w - 4$. At $w = 1$: single gradients $2$ and $-6$; average $= -2 = \nabla J(1) = 2(1) - 4$ ✓.

**2. Full-batch optimum.** $\nabla J = 2w - 4 = 0 \Rightarrow w^* = 2$ — the mean of $\{0, 4\}$ (the MLE of section 04 for known scale).

**3. Why the expectations are equal.** Sampling $I$ uniformly makes $\mathbb{E}[\nabla L_I] = \frac1n\sum_i \nabla L_i$ (average of values weighted by equal probabilities), and linearity turns that sum into $\nabla J$.

**4. $\eta = 0.01$ instead of $0.1$.** (a) Ten times slower progress — many more epochs to approach $w = 3$; (b) the residual bounce around the optimum shrinks (noise floor scales with $\eta$) — steadier but cheaper per step.

**5. Dataset/batch for 64,000 steps per epoch.** $\text{steps} = n/|B|$: e.g. batch $64 \Rightarrow n = 64{,}000 \times 64 = 4{,}096{,}000$; batch $256 \Rightarrow n = 16{,}384{,}000$ (any pair with $n = 64{,}000\,|B|$).

**6. Escaping a saddle.** At a saddle the deterministic gradient is zero (no descent direction), but SGD's zero-mean noise has nonzero variance — random kicks push iterates off the saddle along escaping directions.

## 05-regularization.md

**1. L2 objective and gradient.**

$$
J_{\text{reg}}(w, b) = \frac{1}{n}\sum_{i=1}^n \big(y_i - (wx_i + b)\big)^2 + \frac{\lambda}{2}w^2,
\qquad
\frac{\partial J_{\text{reg}}}{\partial w} = -\frac{2}{n}\sum_{i=1}^n x_i\big(y_i - wx_i - b\big) + \lambda w
$$

**2. Decay-only dynamics.** With no data gradient: $\theta_k = (1 - \eta\lambda)^k \theta_0$ — geometric decay to $0$ since $0 < 1 - \eta\lambda < 1$ (this is exactly the shrinkage factor of chapter 02, harnessed).

**3. Ridge with $X^\top X = 10$, $X^\top y = 20$.** $w = 20/(10 + \lambda)$: $\lambda = 0 \Rightarrow 2$; $\lambda = 5 \Rightarrow 20/15 = 4/3 \approx 1.33$.

**4. Soft-thresholding with $\lambda = 3$.** $a = 1$: $|a| \le \lambda \Rightarrow w^* = 0$. $a = 5$: $5 - 3 = 2 \Rightarrow w^* = 2$. $a = -2$: $|-2| \le \lambda \Rightarrow w^* = 0$.

**5. Prior and sparsity.** **Laplace** prior (density $\propto e^{-|\theta|/b}$): its spike at zero and heavy tails let the posterior mode land exactly at $0$ for small coefficients — unlike the Gaussian, whose smooth peak only shrinks.

**6. Both errors rising with $\lambda$.** **Underfitting** (high-bias regime): the penalty dominates and the model can no longer fit even the training data — decrease $\lambda$ (chapter 01's table: at $\lambda = 5$ the coefficient was halved).

## 06-constrained-optimization.md

**1. Lagrangian and stationarity for $\min x^2 + 4y^2$ s.t. $x + y = 1$.**

$$
\mathcal{L} = x^2 + 4y^2 + \lambda(x + y - 1),\qquad
2x + \lambda = 0,\quad 8y + \lambda = 0,\quad x + y = 1
$$

**2. Solve.** $x = -\lambda/2$, $y = -\lambda/8$; constraint: $-\lambda/2 - \lambda/8 = -5\lambda/8 = 1 \Rightarrow \lambda = -8/5$. Then

$$
x = \tfrac45 = 0.8,\qquad y = \tfrac15 = 0.2,\qquad \lambda = -1.6
$$

(minimum value $f = 0.64 + 4(0.04) = 0.8$; check $\nabla f = (1.6, 1.6) = -\lambda(1,1)$ ✓).

**3. Shadow price.** On $x + y = 2 + \varepsilon$: optimum $(1 + \varepsilon/2)^2 \times 2$, value $2 + 2\varepsilon + \varepsilon^2/2$; $\frac{d}{d\varepsilon}\big|_0 = 2 = -\lambda$ with $\lambda = -2$ from the chapter — the multiplier equals the sensitivity up to the convention's sign.

**4. $\min \tfrac12 w^2$ s.t. $w \ge 3$.** Unconstrained $w = 0$ infeasible ⇒ active constraint $w^* = 3$; stationarity $w + \mu(-1) = 0 \Rightarrow \mu = 3 \ge 0$ ✓; complementary slackness: $h(w^*) = 3 - 3 = 0$ with $\mu = 3$ — product $0$ ✓.

**5. Stationarity in $b$ (SVM).** $\mathcal{L}$'s term in $b$: $-b\sum_i \alpha_i y_i$; $\partial\mathcal{L}/\partial b = -\sum_i \alpha_i y_i = 0 \Rightarrow \sum_i \alpha_i y_i = 0$.

**6. Feasible unconstrained minimum.** The constraint is already satisfied there, so take $\lambda = 0$: stationarity reduces to $\nabla f = 0$ — the constrained and unconstrained solutions coincide (a check: $\lambda = 0$ means the constraint wasn't binding).
