# Kullback–Leibler Divergence

## Why It Matters for AI

Cross-entropy contains two things: the truth's own uncertainty ($H(p)$) and the *mismatch* between truth and model. **KL divergence** is that second part alone — the standard way to say "how far is $q$ from $p$?" for probability distributions. It scores training progress, defines distillation, penalizes drift from a reference policy, and appears in nearly every probabilistic model's objective.

## Definition

$$
\boxed{\;D_{\mathrm{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}
= \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]
= H(p, q) - H(p)\;}
$$

The last equality is chapter 02's decomposition rearranged: KL = cross-entropy minus entropy. Both distributions must live on the same outcomes; conventions: $p_i > 0$ terms only ($0\log\frac{0}{q} = 0$), and if $q_i = 0$ wherever $p_i > 0$, then $D_{\mathrm{KL}} = +\infty$ (chapter 02's exploded cross-entropy, isolated).

**Two readings:**

1. **Expected log-likelihood ratio** — how much $p$ favors each outcome over $q$, averaged under $p$: $\mathbb{E}_p[\log p - \log q]$.
2. **Excess code length** — the average extra nats per symbol when coding $p$-data with $q$'s ideal code instead of $p$'s (Shannon's theorem from chapter 01, compared against itself).

## Worked Example (Same as Chapter 02)

$p = [0.7, 0.2, 0.1]$, $q = [0.6, 0.3, 0.1]$:

$$
D_{\mathrm{KL}}(p \| q) = 0.7\ln\frac{0.7}{0.6} + 0.2\ln\frac{0.2}{0.3} + 0.1\ln\frac{0.1}{0.1}
$$

$$
= 0.7(0.1542) + 0.2(-0.4055) + 0 = 0.1079 - 0.0811 = 0.0268 \text{ nats}
$$

matching $H(p,q) - H(p) = 0.8286 - 0.8018 = 0.0268$ ✓ (verified numerically).

**Reverse direction, same pair:**

$$
D_{\mathrm{KL}}(q \| p) = 0.6\ln\frac{0.6}{0.7} + 0.3\ln\frac{0.3}{0.2} + 0.1\ln\frac{0.1}{0.1}
= -0.0925 + 0.1216 = 0.0291 \text{ nats}
$$

$0.0268 \ne 0.0291$ — **KL is asymmetric.** (Both small here because the distributions are close; the asymmetry is structural, not a rounding artifact.)

```python
from math import log
def kl(p, q):
    return sum(pi * log(pi / qi) for pi, qi in zip(p, q) if pi > 0)
print(kl([0.7,0.2,0.1], [0.6,0.3,0.1]))  # 0.0268
print(kl([0.6,0.3,0.1], [0.7,0.2,0.1]))  # 0.0291
```

## Properties

1. **Non-negative, and zero only at equality:** $D_{\mathrm{KL}}(p\|q) \ge 0$ with equality iff $p = q$ (**Gibbs' inequality**).

   *Proof sketch (Jensen):* for concave $\ln$ (section 00),
   $$
   D_{\mathrm{KL}}(p\|q) = -\,\mathbb{E}_p\!\left[\ln \frac{q(x)}{p(x)}\right]
   \;\ge\; -\ln \mathbb{E}_p\!\left[\frac{q(x)}{p(x)}\right]
   = -\ln \sum_x q(x) = -\ln 1 = 0
   $$
   with equality in Jensen iff $q/p$ is constant — i.e., $q = p$.

2. **Asymmetric:** $D(p\|q) \ne D(q\|p)$ (worked example) — so it is not a metric; triangle inequality also fails. It is a *divergence*.
3. **Direction matters — forward vs reverse KL:**
   - $D(p\|q)$ ("forward", weights $p$): infinite if $q$ gives **zero** to anything $p$ supports — $q$ is forced to **cover** $p$'s mass everywhere (mass-covering).
   - $D(q\|p)$ ("reverse"): infinite if $p$ is zero where $q$ isn't — $q$ may **concentrate** on modes $p$ favors and drop the rest (mode-seeking).
   The two directions encode genuinely different modeling behaviors — the reason variational methods choose theirs deliberately.
4. **Units:** nats (natural log). Adds over independent components: $D(p_X p_Y \| q_X q_Y) = D(p_X\|q_X) + D(p_Y\|q_Y)$.
5. **Sensitive to tails:** a small probability placed wrongly contributes $p \ln(p/q)$ with $q$ tiny — divergence grows like $\ln(1/q)$ times $p$; near-zero mismatches on likely events dominate.

## Connection to AI

- **MLE = minimizing empirical KL.** Fitting $q_\theta$ by maximum likelihood on data distributed as $\hat p$ is (up to $H(\hat p)$) minimizing $D_{\mathrm{KL}}(\hat p \| q_\theta)$ — statistics' likelihood and information theory's divergence, one operation (chapter 02's identity).
- **Knowledge distillation:** a student is trained to match a teacher's *soft* output distribution: minimize $D_{\mathrm{KL}}(\text{teacher} \| \text{student})$ per example — forward KL, so the student must cover all the teacher's probability mass (the softened target supplies information beyond the hard label).
- **KL as a training-time quantity:** monitoring $D_{\mathrm{KL}}$ between a model's outputs before/after an update ("KL penalty" in RL fine-tuning) measures drift from a reference policy — a trust-region idea: keep updates close in KL, keep behavior stable (section 10's orbit).
- **VAEs** bound the log-likelihood with an objective containing $D_{\mathrm{KL}}(q_\theta(z\|x) \| p(z))$ — the divergence between an encoder and a prior (beyond this course's scope; noted as where the symbol reappears).
- **Why "distance" colloquially but not formally:** it reads like a distance, but asymmetry means "the KL from my belief to yours" ≠ "from yours to mine" — always state the direction.

## Common Mistakes

- Reversing the arguments — forward vs reverse KL behave differently (property 3); $D(p\|q)$ weights by $p$.
- Treating it as a metric — no symmetry, no triangle inequality; don't plug it into algorithms needing true distances.
- Comparing KL values across different distributions/alphabets — the value depends on both; only same-$p$ comparisons across different $q$'s rank "closeness to $p$".
- Ignoring $q_i \approx 0$ on $p_i > 0$ events — infinite/large divergence dominates; this is why training clips probabilities and why confident errors are expensive.
- Reporting $D$ in bits when the code used nats (factor $\ln 2$).

## Exercises

1. Compute $D_{\mathrm{KL}}([0.5, 0.5] \,\|\, [0.9, 0.1])$ in nats. Which term dominates, and why?
2. Show directly that $D_{\mathrm{KL}}(p \| p) = 0$ for any $p$.
3. Compute both directions for $p = [0.5, 0.5]$, $q = [0.6, 0.4]$; confirm they differ.
4. What is $D_{\mathrm{KL}}([0.9, 0.1] \,\|\, [1, 0])$? State the general rule it illustrates.
5. Using chapter 02's numbers, confirm $D_{\mathrm{KL}}(p\|q) = H(p, q) - H(p)$ as an arithmetic identity (nats).
6. In knowledge distillation, which KL direction is minimized — $D(\text{teacher}\|\text{student})$ or $D(\text{student}\|\text{teacher})$ — and what does the forward direction force the student to do with probability mass? (One sentence.)
