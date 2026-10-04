# 10 — LLM Mathematics: Solutions

## 01-pipeline-and-next-token.md

**1. Shapes and parameters.** $X_0 \in \mathbb{R}^{17 \times 2048}$; $\mathbf{h}_{17} \in \mathbb{R}^{2048}$ (row 17 of $X_N$); $\boldsymbol{\ell}_{17} \in \mathbb{R}^{50{,}000}$; $P_{17} \in \mathbb{R}^{50{,}000}$ (sums to 1). $W_U$: $50{,}000 \times 2048 = \mathbf{102{,}400{,}000}$ $\approx$ **102.4M parameters**.

**2. Uniform on two tokens.** Equal logits: $e^{0} = e^{0}$ ⇒ $P = (0.5, 0.5)$. At least $1/V$ somewhere: the probabilities sum to $1$ across $V$ entries, so by pigeonhole some entry is $\ge 1/V$ (and softmax's strictly positive numerators $e^{\ell_j} > 0$ make *every* entry positive — no token is ever assigned exact zero on finite logits).

**3. Logits $(3, 0)$.** $e^3 = 20.086$, $e^0 = 1$, sum $21.086$: $P = (0.953,\ 0.047)$; $-\ln(0.953) \approx 0.049$ nats — even the top prediction, when assigned $95\%$, still costs $\approx 0.05$ nats if correct.

**4. Why the mask legitimizes the conditional.** The $-\infty$ mask zeroes attention to future positions, so position $t$'s hidden state (and hence logits) is a function of $w_{\le t}$ only — the emitted distribution *cannot* depend on $w_{t+1}$ or later, which is exactly what $P(w_{t+1} \mid w_{\le t})$ claims.

**5. Tied vs untied.** Separate $W_U$: another $50{,}000 \times 2048 = 102.4$M parameters *in addition to* $E$ (total for both: 204.8M). Tied: $W_U := E$ — **0 extra** parameters (the unembedding reuses the embedding matrix; gradients from the output layer flow into $E$ as well).

**6. Distributions per batch item.** One per position: **512** distributions (and 511–512 loss terms depending on convention) per sequence — chapter 02's sum runs over all of them, all from one forward+backward pass.

## 02-training-objective.md

**1. Factorization and NLL.**
$$
P(w_1, w_2, w_3) = P(w_1)\,P(w_2 \mid w_1)\,P(w_3 \mid w_1, w_2)
$$
$$
\text{NLL} = -\ln P(w_1) - \ln P(w_2 \mid w_1) - \ln P(w_3 \mid w_1, w_2),\qquad
\mathcal{L} = \frac{\text{NLL}}{3}
$$

**2. Loss for probabilities $(0.5, 0.5)$.**
$$
\mathcal{L} = -\tfrac{1}{2}[\ln 0.5 + \ln 0.5] = -\ln 0.5 = \ln 2 \approx \mathbf{0.6931} \text{ nats/token}
$$

**3. Probabilities $(0.9, 0.9)$.** $\mathcal{L} = -\ln 0.9 \approx \mathbf{0.1054}$ nats/token. The first corpus is the more surprising one: $0.6931 - 0.1054 = \mathbf{0.5877}$ nats/token more surprise per token (the model assigns its tokens well in the second case).

**4. Cross-entropy with a one-hot target.** Target at position $t$: one-hot with $1$ at $w_{t+1}$; section 06's cross-entropy collapses:
$$
H(\delta_{w_{t+1}}, P_\theta(\cdot \mid w_{\le t})) = -\sum_j \delta_j \ln P_\theta(j \mid w_{\le t}) = -\ln P_\theta(w_{t+1} \mid w_{\le t}) \;✓
$$

**5. Terms in the mean.** $8 \times 99 = \mathbf{792}$ log-probability terms (each length-100 sequence contributes 99 predictions — the token has no prediction before it starts).

**6. Teacher forcing.** Training conditions each prediction on the **true** prefix; generation conditions on the model's **own previously generated** tokens — the source of the train/generation mismatch named in the chapter.

## 03-perplexity.md

**1. Losses to PPL.** $\mathcal{L} = 0 \Rightarrow e^0 = \mathbf{1}$; $\mathcal{L} = \ln 4 \Rightarrow \mathbf{4}$; $\mathcal{L} = 3.0 \Rightarrow e^3 \approx \mathbf{20.09}$.

**2. PPL $= 100$.** Average log-probability $= -\ln 100 \approx \mathbf{-4.605}$ nats/token; geometric-mean true-token probability $= 1/100 = \mathbf{0.01}$.

**3. Uniform over $V = 50{,}000$.** Loss $= \ln 50{,}000 \approx \mathbf{10.820}$ nats/token (the maximum-entropy value, section 06); PPL $= \mathbf{50{,}000}$ — the vocabulary size, the "uninformed" landmark.

**4. Best possible PPL at $H = 1.4$.** Ideal model: $\mathcal{L} = H = 1.4$ ⇒ PPL $= e^{1.4} \approx \mathbf{4.055}$. Model at PPL $e^{1.7}$: $\mathcal{L} = 1.7$ — it sits $1.7 - 1.4 = \mathbf{0.3}$ nats/token **above the floor**, i.e., that entire gap is avoidable error (section 04/06's decomposition: model's cross-entropy $=$ entropy + KL).

**5. A vs B.** Lower PPL is better: **A** (25 vs 30). Difference: $\ln 30 - \ln 25 = \ln(1.2) \approx \mathbf{0.182}$ nats/token — B is surprised $0.182$ nats more per token (on $1000$ tokens: $182$ nats total $\Rightarrow e^{182}\times$ likelihood ratio in A's favor).

**6. Why bits-per-byte is fairer.** The tokenizer decides how many prediction *steps* a byte spans — bits-per-byte divides total information by raw bytes, removing the tokenization-dependent denominator so different vocabularies/tasks become comparable.

## 04-sampling-and-temperature.md

**1. Logits $(1, 0)$.** $T = 0.5$: scaled $(2, 0)$ ⇒ $(\frac{7.389}{8.389}, \frac{1}{8.389}) = (\mathbf{0.881},\ \mathbf{0.119})$. $T = 2$: scaled $(0.5, 0)$ ⇒ $(\frac{1.649}{2.649}, \frac{1}{2.649}) = (\mathbf{0.622},\ \mathbf{0.378})$. Ranking preserved: first entry larger in both.

**2. Limits.** $T \to 0^+$: one-hot on the argmax, $H \to \mathbf{0}$ (certain). $T \to \infty$: uniform, $H \to \ln V$ (maximum entropy, section 06) — for a $V$-token vocab, $H = \ln V$.

**3. Top-$p = 0.8$ on $(0.4, 0.3, 0.2, 0.1)$.** Cumulative: $0.4$, $0.7$, $0.9 \ge 0.8$ — keep the first **three** (sum $0.9$); renormalize: $(\mathbf{0.444},\ \mathbf{0.333},\ \mathbf{0.222})$ — the $0.1$ token is excluded.

**4. Raising $T$ raises entropy.** By the limits: $T \to 0$ gives $H = 0$ and $T \to \infty$ gives $H = \ln V$; raising $T$ shrinks logit gaps monotonically toward uniform — mass spreads out, so $H(P_T)$ increases continuously between those endpoints.

**5. Top token at $T = 0.5$.** Probability $0.864$; self-information if correct: $-\ln 0.864 \approx \mathbf{0.146}$ nats — confident correctness is cheap (chapter 02's cost function).

**6. Why temperature can't improve PPL.** PPL measures the model's own $T = 1$ conditional probabilities on held-out data; temperature only reshapes the *sampling* distribution after training — the model's likelihoods, and hence PPL, are untouched.
