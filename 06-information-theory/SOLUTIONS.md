# 06 — Information Theory: Solutions

## 01-entropy.md

**1. Entropies in nats.**
$$
H([0.5,0.5]) = \ln 2 \approx 0.6931,\qquad
H([0.9,0.1]) \approx 0.3251,\qquad
H([0.25,0.25,0.25,0.25]) = \ln 4 \approx 1.3863
$$

**2. Which is higher, $[0.9,0.1]$ or $[0.6,0.4]$?** $[0.6,0.4]$: $0.6730 > 0.3251$ nats. Intuition: entropy measures *skew* — $[0.9,0.1]$ is far from uniform (near-certain), $[0.6,0.4]$ is closer to a fair split (more uncertainty).

**3. Unit conversion.** $2$ nats $= 2/\ln 2 \approx 2.885$ bits; $3$ bits $= 3\ln 2 \approx 2.079$ nats.

**4. Fair 8-sided die.** $\log_2 8 = 3$ bits. In nats: $\ln 8 \approx 2.0794$; conversion check: $3 \times \ln 2 = 2.0794$ ✓.

**5. Why $H = 0$ only for deterministic $p$.** Each term $-p_i\log p_i \ge 0$ (since $\log p_i \le 0$), and it is zero only when $p_i \in \{0,1\}$ — so the sum vanishes iff exactly one outcome has probability 1.

**6. Two independent fair coins.** Additivity: $H(X,Y) = H(X) + H(Y) = 1 + 1 = 2$ bits. Direct: 4 joint outcomes, each $\tfrac14$: $-4 \cdot \tfrac14 \log_2 \tfrac14 = 2$ bits ✓.

## 02-cross-entropy.md

**1. $H(p,q)$ for $p = [1,0]$, $q = [0.9,0.1]$.** Only $p_1$ contributes: $-\ln 0.9 \approx 0.1054$ nats.

**2. $p = [1,0]$, $q = [0.5,0.5]$.** $-\ln 0.5 \approx 0.6931$ nats. The first $q$ ($0.9$ on the certain event) is better — smaller cross-entropy — because it puts more probability exactly where $p$ puts its mass; the cost is the model's probability on the *real* event.

**3. Decomposition check.** Direct: $0.7\ln\tfrac{0.7}{0.6} + 0.2\ln\tfrac{0.2}{0.3} + 0.1\ln\tfrac{0.1}{0.1} = 0.1079 - 0.0811 + 0 = 0.0268$ nats $= 0.8286 - 0.8018$ ✓.

**4. Binary cross-entropy.** $y = 1$: $-\ln 0.8 \approx 0.2231$ nats. $y = 0$: $-\ln(1-0.8) = -\ln 0.2 \approx 1.6094$ nats. The confident wrong prediction ($q = 0.8$ for label $0$) costs $\approx 7.2\times$ more — cross-entropy punishes confident errors hardest.

**5. Why this loss = MLE.** The averaged per-example NLL is exactly $-\log$ likelihood of the categorical model (section 04), and for fixed model class its minimizer sets $q$ to the empirical frequencies — MLE and cross-entropy minimization are the same optimization problem.

**6. $q_t = 10^{-3}$.** $-\ln 10^{-3} = \ln 1000 \approx 6.908$ nats $\approx 9.97$ bits (one rare-but-wrong prediction dominates an epoch's average).

## 03-kullback-leibler-divergence.md

**1. $D_{\mathrm{KL}}([0.5,0.5] \,\|\, [0.9,0.1])$.**
$$
0.5\ln\tfrac{0.5}{0.9} + 0.5\ln\tfrac{0.5}{0.1} = -0.2939 + 0.8047 = 0.5108 \text{ nats}
$$
Dominated by the **second** term: it covers the outcome where $q$ ($0.1$) underestimates $p$ ($0.5$) — underestimating a likely event is expensive ($\ln\frac{p}{q}$ large).

**2. $D_{\mathrm{KL}}(p \| p) = 0$.** Every term is $p_i \ln\frac{p_i}{p_i} = p_i \ln 1 = 0$; the sum is 0.

**3. Both directions for $[0.5,0.5]$ vs $[0.6,0.4]$.**
$$
D(p\|q) = 0.5\ln\tfrac{0.5}{0.6} + 0.5\ln\tfrac{0.5}{0.4} = -0.0912 + 0.1116 = 0.0204
$$
$$
D(q\|p) = 0.6\ln\tfrac{0.6}{0.5} + 0.4\ln\tfrac{0.4}{0.5} = 0.1094 - 0.0893 = 0.0201
$$
$0.0204 \ne 0.0201$ — asymmetric ✓.

**4. $D_{\mathrm{KL}}([0.9,0.1] \,\|\, [1,0])$.** $+\infty$: the second component has $p_2 = 0.1 > 0$ but $q_2 = 0$, so the term $0.1\ln\frac{0.1}{0}$ diverges. **Rule:** forward KL is infinite whenever $q$ assigns zero probability to anything $p$ supports ($q$ must cover $p$'s support).

**5. Arithmetic identity.** $H(p,q) - H(p) = 0.8286 - 0.8018 = 0.0268$ nats, matching the direct KL computation of chapter 02's numbers ✓.

**6. Distillation direction.** $D_{\mathrm{KL}}(\text{teacher} \| \text{student})$ — forward KL: the student must place nonzero probability mass on every outcome the teacher deems possible (mass-covering; a zero would make the divergence infinite).

## 04-conditional-entropy.md

**1. Both slices deterministic at `a`.** $H(Y \mid X = x) = 0$ for each $x$, so $H(Y \mid X) = 0$. But $Y$ is constant regardless of $X$, so $H(Y) = 0$ too: $X$ is *useless* ($I(X;Y) = 0$) — the zero uncertainty comes from $Y$'s determinism, not from $X$'s information.

**2. Average of slice entropies.** $H(Y \mid X) = \tfrac12(1) + \tfrac12(3) = 2$ nats.

**3. $Y = X$ binary uniform.** $H(Y \mid X) = 0$ (each slice is a point mass). $H(X, Y) = H(X) = \ln 2 \approx 0.6931$ nats — the pair is fully determined by $X$ alone.

**4. Chain-rule check.** $H(X \mid Y) = H(X,Y) - H(Y) = 1.1935 - 0.6931 = 0.5004$; then $H(Y) + H(X \mid Y) = 0.6931 + 0.5004 = 1.1935$ nats ✓ (equals $H(X) + H(Y \mid X)$ — the two peels agree).

**5. Independent case.** $p(y \mid x) = p(y)$ for all $x$, so $H(Y \mid X) = \sum_x p(x) H(Y) = H(Y)$ — conditioning changes nothing.

**6. Test cross-entropy $0.60$ vs $H(Y\mid X) = 0.55$.** The $0.05$-nat gap is the model's **avoidable error** — excess over the Bayes-optimal log-loss. A score of $0.50$ (below the conditional entropy) would be impossible on fresh data: it signals leakage, evaluation on training data, or a wrong entropy estimate.

## 05-mutual-information.md

**1. Independence, three forms.** $H(Y) - H(Y \mid X) = H(Y) - H(Y) = 0$; $H(X) - H(X \mid Y) = 0$ by symmetry of the same argument; $D_{\mathrm{KL}}(p(x,y) \| p(x)p(y)) = 0$ because the joint *equals* the product of marginals under independence.

**2. Deterministic relation.** $H(Y \mid X) = 0$, so $I(X;Y) = H(Y) - 0 = 2$ nats — the upper bound $I \le \min(H(X), H(Y))$ forces $H(X) \ge 2$ as well.

**3. Chapter's table via form 2.** $H(X \mid Y) = H(X,Y) - H(Y) = 1.1935 - 0.6931 = 0.5004$; $I(X;Y) = H(X) - H(X \mid Y) = 0.6931 - 0.5004 = 0.1927$ nats ✓ (all three forms agree).

**4. Uniform joint on 4 outcomes.** $p(x,y) = \tfrac14 = \tfrac12 \cdot \tfrac12 = p(x)p(y)$ — the joint already factors, so $I(X;Y) = 0$: no shared information, pure independence.

**5. $H(Y) = 1.0$, $H(Y \mid X) = 0.7$.** $I(X;Y) = 0.3$ nats: a predictor using $X$ achieves an expected log-loss $0.3$ nats per example lower than one that ignores $X$ — exactly the conditional-entropy reduction $X$ buys.

**6. Feature triage.** Feature A is useless: $I(A;Y) = 0$ means knowing $A$ does not reduce uncertainty about $Y$ ($H(Y \mid A) = H(Y) = 0.9$ nats). For B: $H(Y \mid B) = H(Y) - I(B;Y) = 0.9 - 0.3 = 0.6$ nats.
