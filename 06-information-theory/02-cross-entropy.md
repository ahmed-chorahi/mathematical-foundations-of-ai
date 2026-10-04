# Cross-Entropy

## Why It Matters for AI

Ask a model for a probability — it answers with $q$. The truth (or the data) has distribution $p$. **Cross-entropy** $H(p, q)$ measures the average cost of describing reality using $q$'s code: how surprised, on average, the events drawn from $p$ are *according to $q$*. It equals the entropy exactly when $q = p$ and is worse otherwise. This is the loss behind logistic regression, image classifiers, and language models — the single most-used objective in machine learning.

## Definition

For distributions $p$ (truth) and $q$ (model) on the same outcomes:

$$
\boxed{\;H(p, q) = \mathbb{E}_{x \sim p}\big[-\log q(x)\big] = -\sum_i p_i \log q_i\;}
$$

Same shape as entropy, but the **expectation weights come from $p$ while the log comes from $q$**: real events, model's beliefs. (When $q = p$ the two definitions coincide: $H(p,p) = H(p)$.)

**Worked example.** Truth $p = [0.7, 0.2, 0.1]$; model $q = [0.6, 0.3, 0.1]$:

$$
H(p, q) = -(0.7\ln 0.6 + 0.2\ln 0.3 + 0.1\ln 0.1)
$$

$$
0.7\ln 0.6 \approx -0.3576,\quad 0.2\ln 0.3 \approx -0.2408,\quad 0.1\ln 0.1 \approx -0.2303
$$

$$
H(p, q) \approx 0.8286 \text{ nats}
$$

Compare the true entropy from chapter 01: $H(p) \approx 0.8018$ nats. The model's mismatch costs $0.8286 - 0.8018 = 0.0268$ nats per event — that gap has a name (KL divergence, next chapter).

```python
from math import log
def cross_entropy(p, q):
    return -sum(pi * log(qi) for pi, qi in zip(p, q) if pi > 0)
print(cross_entropy([0.7, 0.2, 0.1], [0.6, 0.3, 0.1]))  # 0.8286 nats
```

## The Decomposition (Central Identity)

$$
\boxed{\;H(p, q) = H(p) + D_{\mathrm{KL}}(p \,\|\, q)\;}
$$

*Proof (one line, algebra):*

$$
-\sum_i p_i \log q_i = -\sum_i p_i \log p_i + \sum_i p_i \log\frac{p_i}{q_i}
\quad\Longleftrightarrow\quad
H(p, q) = H(p) + \sum_i p_i \ln\frac{p_i}{q_i}
$$

add and subtract $\sum p_i \log p_i$. Since KL $\ge 0$ (chapter 03): **cross-entropy is minimized, over all $q$, only at $q = p$**, where it equals the irreducible $H(p)$. This is the sense in which "cross-entropy training recovers the truth if the model class allows it".

## As the Training Loss

**Classification.** Each training example has a true label $t$ (one-hot — the empirical distribution concentrated on one class). The per-example cross-entropy between that one-hot $p$ and the model's predicted probabilities $q$ collapses to:

$$
H(p_{\text{one-hot}(t)},\, q) = -\log q_t
$$

(all terms have $p_i = 0$ except the true class). Average over the dataset:

$$
\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^n \log q_\theta(t_i \mid x_i)
$$

— exactly the **negative log-likelihood** of the categorical model from section 04, written as a cross-entropy. Maximizing likelihood = minimizing cross-entropy; the statistics and information-theory readings of the same objective.

**Binary case** (label $y \in \{0,1\}$, model $q$ = predicted prob of 1):

$$
\text{BCE} = -\big[y \log q + (1-y)\log(1-q)\big]
$$

one-hot cross-entropy with two classes. Logistic regression's loss (section 07) is this function; its smooth convex shape was verified in section 05 (composition rule: $\ln(1 + e^{-z})$ convex in the logit $z$).

**PyTorch-style naming:** the layer computes `log_softmax(logits)`; the loss is `nll_loss` (negative log likelihood); the composite is `cross_entropy` — all three names describe this chapter's function on the same numbers.

## Properties

- $H(p, q) \ge H(p)$, equality iff $p = q$ (via the decomposition).
- **Asymmetric:** $H(p,q) \ne H(q,p)$ in general — $p$ supplies the weights, $q$ the logs. Swapping arguments changes the meaning entirely (chapter 03 quantifies both directions).
- **Can exceed $\ln k$?** Yes — if $q$ assigns tiny probability to an outcome $p$ favors, $-\log q$ explodes. An overconfident wrong prediction ($q_t = 10^{-6}$) costs $\approx 13.8$ nats on one example — the reason cross-entropy punishes confident errors so hard, and the reason training is sensitive to a model's worst mistakes.
- **Not a distribution-gap by itself:** $H(p, q)$ mixes the irreducible $H(p)$ with the mismatch; subtracting $H(p)$ isolates the pure gap between $p$ and $q$ — that is KL divergence (chapter 03). Cross-entropy is a cost, not a distance.

## Connection to AI

- **Every softmax classifier** ends in this chapter: logits $\to$ softmax $\to$ $-\log q_t$. Section 08 (neural nets) and section 09 (transformers) inherit it unchanged.
- **Language modeling** is cross-entropy over the vocabulary at every position, summed over tokens (section 10) — LLM quality is reported as this loss (and its exponential, perplexity).
- **Label smoothing, focal loss, knowledge distillation** all modify this objective (soften $p$, reweight examples, or soften targets) — each interpretable as changing $p$ or the weighting inside $H(p,q)$.
- **Optimization link:** convexity chapter 05 verified the binary loss is convex in the logit — for generalized linear models, minimizing $H(p, q)$ has the global-optimum guarantee.
- **Calibration:** a well-trained model should have $q$ match empirical frequencies — cross-entropy is strictly proper: its expected value is minimized by truthful probabilities (the decomposition *is* the proof).

## Common Mistakes

- Swapping $p$ and $q$: $H(q, p)$ weights by the model's beliefs — a different quantity (KL's other direction). **$p$ always supplies the weights.**
- Summing $-\log q_c$ over *all* classes instead of the true class: that is $H(\text{uniform}, q)$-shaped and is minimized by predicting uniform — labels would be ignored. Per-example loss reads **one** probability: $-\log q_t$.
- Computing $\log 0$: floating-point models clip $q$ (or use `log_softmax` for stability) — an implementation detail with mathematical content (the $0\log0$ convention).
- Confusing base: losses in code are nats; reported "bits per character" (a language-model metric) divides by $\ln 2$.
- Expecting $H(p,q) < H(p)$ for a good model — impossible; $q = p$ is the floor. A *model's* cross-entropy below the *data's* entropy indicates leakage or estimation error.

## Exercises

1. Compute $H(p, q)$ in nats for $p = [1, 0]$ (certain outcome 1), $q = [0.9, 0.1]$.
2. Same $p = [1, 0]$, but $q = [0.5, 0.5]$: compute; which $q$ is better for this $p$, and why is that visible without formulas?
3. Verify the decomposition with the chapter's numbers: $0.8286 = 0.8018 + 0.0268$ — compute $D_{\mathrm{KL}} = 0.7\ln\tfrac{0.7}{0.6} + 0.2\ln\tfrac{0.2}{0.3} + 0.1\ln\tfrac{0.1}{0.1}$ directly and confirm it matches the gap.
4. BCE for: $y = 1, q = 0.8$; and $y = 0, q = 0.8$. Which error hurts more, and by how much (nats)?
5. Why does minimizing $\mathcal{L} = -\frac1n\sum \log q_\theta(t_i|x_i)$ equal MLE? (One sentence referencing section 04's log-likelihood.)
6. A model outputs $q_t = 10^{-3}$ on a true label: what is the per-example loss in nats? In bits? (Use $\ln 1000 \approx 6.908$.)
