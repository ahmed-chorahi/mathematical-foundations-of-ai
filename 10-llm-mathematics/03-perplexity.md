# Perplexity

## Why It Matters for AI

Training loss is in nats — precise but unreadable ("is 3.2 good?"). **Perplexity** exponentiates it into a quantity with a direct reading: the number of equally likely choices the model effectively faces at each token. It is the standard single-number summary of a language model's predictive quality — and it is nothing more than $\exp(\text{cross-entropy})$, built entirely from sections 06's entropy.

## Definition

For average per-token cross-entropy $\mathcal{L}$ in **nats**:

$$
\boxed{\;\text{PPL} = e^{\mathcal{L}} = \exp\!\left(-\frac{1}{T}\sum_{t} \ln P_\theta(w_{t+1}\mid w_{\le t})\right) = \left(\prod_{t} \frac{1}{P_\theta(w_{t+1}\mid w_{\le t})}\right)^{1/T}\;}
$$

The rightmost form: the **geometric mean of the inverse probabilities** the model assigned the true tokens — "how many equally-probable options, on average, would have given the same score?" Each true token had probability $1/\text{PPL}$ on average (geometric mean).

**Worked example (from chapter 02).** $\mathcal{L} = 0.8574$ nats/token:

$$
\text{PPL} = e^{0.8574} \approx \mathbf{2.36}
$$

Read: the model is as uncertain as facing ~2.36 equally likely next tokens at each step of that sequence. (Computed directly: geometric mean of $1/0.6 = 1.667$ and $1/0.3 = 3.333$: $\sqrt{1.667 \times 3.333} = \sqrt{5.556} = 2.357$ ✓.)

**Second worked example.** Uniform loss $\mathcal{L} = \ln 2 \approx 0.6931$: PPL $= 2$ — a fair coin's worth of uncertainty. Loss $\mathcal{L} = 2.0$: PPL $= e^2 \approx 7.39$, average true-token probability $1/7.39 \approx 0.135$.

```python
import math
print(math.exp(0.8574))    # 2.358
print(math.exp(2.0))       # 7.389
```

## Range and Landmarks

$$
1 \;\le\; \text{PPL} \;\le\; V
$$

| PPL | meaning |
|---|---|
| $1$ | perfect confidence ($\mathcal{L} = 0$): every true token had probability 1 |
| $\text{PPL} = k$ | "as uncertain as $k$ equal choices" (the canonical reading) |
| $V$ | uniform guessing over the vocabulary ($\mathcal{L} = \ln V$ — entropy's maximum, section 06) |
| $> V$ | worse than uniform on this data — systematically miscalibrated (possible on out-of-domain text) |

**Bits form.** With loss in bits: $\text{PPL} = 2^{\mathcal{L}_{\text{bits}}}$ (same quantity, other log base — section 06's conversion; $\text{PPL} = e^{\mathcal{L}_{\text{nats}}} = 2^{\mathcal{L}_{\text{nats}}/\ln 2}$).

**Connection to entropy — the deep identity.** If the model *were* the true data distribution $p$, its cross-entropy would equal $H(p)$ (section 06, chapter 02: $H(p, p) = H(p)$), so

$$
\text{PPL} = e^{H(p)} = \exp\big(H(W_{t+1} \mid W_{\le t})\big)
$$

— the perplexity of the *ideal* model equals the exponential of the corpus's conditional entropy: irreducible ambiguity (section 04's $H(Y \mid X)$) sets the floor. No model beats $e^{H}$ on held-out data from the same source (chapter 02 of section 06's floor argument); the gap between a trained model and that floor is its avoidable error.

## Comparisons: What PPL Can and Cannot Say

**Valid:**

- Same model family, same tokenizer, same domain: lower PPL on held-out data = better predictor (fewer nats/token = the objective improved — and by section 04's estimation discipline, on *unseen* data).
- Tracking a training run: PPL falling = the model's conditionals sharpening on truth.

**Invalid / careful:**

- **Across tokenizers/vocabularies:** PPL counts *tokens*; a tokenizer that splits "unerasable" into 3 tokens gives 3 chances with smaller decisions vs one big one — PPLs from different tokenizations measure different tasks. (The honest cross-model currency is bits per byte/character: $\text{PPL}^{\,1/\text{tokens}}$ converted through the tokenizer's expansion rate.)
- **Across domains:** a news-trained model on medical text has high PPL — difficulty of the data, not only quality of the model.
- **As a goal in itself:** optimizing PPL to the exclusion of downstream behavior overfits the likelihood race — training $\mathcal{L}$ can fall while usefulness stalls (section 08's "the loss is a proxy" honesty); also: low PPL requires *calibration*, not just ranking.

**Worked comparison.** Model A: $\mathcal{L}_A = 1.8$ → PPL $= 6.05$. Model B: $\mathcal{L}_B = 2.1$ → PPL $= 8.17$. B assigns the data $2.1 - 1.8 = 0.3$ nats/token more surprise — on a 1000-token document that compounds to $e^{300} \approx 10^{130}$ times lower likelihood for B vs A (log-likelihood differences multiply intoastronomical ratios — why *log* scale is the honest one to report; PPL is the readable projection of it).

## Connection to AI

- **The number in papers and dashboards:** "WikiText PPL 20" etc. — this formula, evaluated on a held-out set.
- **Cross-entropy ↔ perplexity ↔ bits** are three units of the same measurement (nats/token, exponentiated, base-2) — converting fluently is table stakes for reading results.
- **Temperature evaluation:** PPL is measured at temperature $1$ (the model's own distribution); post-hoc sharpening doesn't change the model's likelihood (chapter 04: temperature reshapes sampling, not the trained conditionals).
- **Early stopping / model selection** for LLMs: held-out PPL is the generalization estimate — section 04's estimation applied to $\theta$ (this course's final convergence of themes: sample, estimate, compare).
- **Scaling laws** (power-law PPL improvements vs compute/data/parameters) are empirical regularities in this metric — the field's planning tool, measured exactly here.

## Common Mistakes

- Reporting training PPL as the result — always held-out (section 04's optimism bias; section 08's checklist).
- Comparing PPL across different tokenizers/vocab sizes — different tasks; convert to bits-per-byte for cross-model claims.
- Treating PPL as linear ("2× better") — PPL differences are exponential in loss; report the loss gap too (0.3 nats/token is the *linear* fact).
- Computing perplexity from *averaged probabilities* instead of averaged *log* probabilities (arithmetic vs geometric mean — the formula uses $\exp$ of mean log; mixing them gives different numbers).
- Forgetting $\text{PPL} \ge 1$ (loss $\ge 0$) — a number below 1 signals a sign error in the loss.
- Using teacher-forced PPL to describe free-running generation quality — they are correlated but distinct (chapter 02's exposure gap).

## Exercises

1. Convert losses to PPL: $\mathcal{L} = 0$, $\mathcal{L} = \ln 4$, $\mathcal{L} = 3.0$.
2. PPL $= 100$: what is the average log-probability per token (nats), and the average true-token probability (geometric mean)?
3. Vocabulary $V = 50{,}000$: what loss gives uniform (uninformed) predictions, and what is that PPL?
4. Using section 06: if the true conditional entropy is $H = 1.4$ nats/token, what PPL is the *best possible* model's perplexity on the same source? What is a model at PPL $= e^{1.7}$ doing in nats/token vs that floor?
5. Model A PPL 25, model B PPL 30 — both tokenized the same way. Which is better on this data, and by how many nats/token? ($\ln 30 - \ln 25$.)
6. Why is bits-per-byte fairer across tokenizers? (One sentence: what the tokenizer changes.)
