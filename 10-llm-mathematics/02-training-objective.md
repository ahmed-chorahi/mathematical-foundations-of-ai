# The Training Objective: Sequence Cross-Entropy

## Why It Matters for AI

How does a model *learn* to produce those distributions? By being scored on them at every position: the training objective is the average negative log-probability the model assigns to the tokens that actually occurred — the cross-entropy of section 06, applied per position and summed over the sequence. This chapter derives the objective from the probability factorization (section 03), shows it *is* the MLE of section 04/06 one more time, and states exactly what teacher forcing does.

## From Factorization to Loss

**Chain rule for probabilities (section 03).** Any sequence probability factors into conditionals:

$$
P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
$$

($P(w_1 \mid w_{<1}) := P(w_1)$ — empty prefix.) This is the probabilistic license for autoregression: the hard joint over sequences is a product of the *same kind* of next-token conditionals the pipeline outputs (chapter 01) — provided the model is causal (section 09's mask), one forward pass at position $t-1$ supplies exactly $P(w_t \mid w_{<t})$.

**Negative log-likelihood of one sequence:**

$$
-\ln P(w_1, \ldots, w_T) = -\sum_{t=1}^{T} \ln P(w_t \mid w_{<t})
$$

**The training objective — mean over positions:**

$$
\boxed{\;\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \ln P_\theta(w_{t+1} \mid w_{\le t})\;}
$$

(there are $T-1$ prediction terms for a length-$T$ sequence; the $1/T$ is a convention — sum-vs-mean only rescales $\eta$'s effective size, but be consistent, section 05's footnote-scale warning).

**Read it three ways — all true:**

1. **Information theory (section 06):** the cross-entropy between the data distribution (one-hot on the actual next token) and the model's conditional, averaged over positions — chapter 02 of section 06, applied $T-1$ times.
2. **Statistics (section 04):** the negative log-likelihood of the autoregressive model — maximizing it *is* MLE; the factorization above is the model's likelihood of the corpus.
3. **Optimization (section 05):** a differentiable scalar in $\theta$; backprop (section 08) through the pipeline of chapter 01 gives $\nabla_\theta \mathcal{L}$; SGD descends.

## Worked Example

Sequence: $(\texttt{the},\ \texttt{cat},\ \texttt{sat})$. Model assigns (from its conditionals):

$$
P(\texttt{cat} \mid \texttt{the}) = 0.6, \qquad
P(\texttt{sat} \mid \texttt{the}, \texttt{cat}) = 0.3
$$

$$
\mathcal{L} = -\frac{1}{2}\big[\ln 0.6 + \ln 0.3\big]
= \frac{0.5108 + 1.2040}{2} = \frac{1.7148}{2} = 0.8574 \text{ nats/token}
$$

(total NLL $1.7148$ nats; per-token mean $0.8574$; perplexity $= e^{0.8574} \approx 2.36$ — chapter 03). Training lowers this by moving $\theta$ so *the actual next tokens* get more probability — at every position simultaneously (the sum runs over all $t$; one backward pass covers every term — section 08's single sweep).

**What the gradient "wants":** for each position, the softmax+CE collapse (section 07) gives error $P_\theta(\cdot \mid w_{\le t}) - \text{one-hot}(w_{t+1})$ at that position's logits — push down the probability of everything that wasn't taken, push up what was, proportional to how wrong the model was. Confident mistakes (chapter 02 of section 06: $-\ln 10^{-3} \approx 6.9$ nats) dominate the update — exactly the penalty structure the curriculum has been building to.

## Teacher Forcing (and the Train/Test Gap)

Training conditions on the **true** prefix $w_{<t}$ — the ground-truth history — never on the model's own generations. This is **teacher forcing**:

- **Why:** every position's term is then a well-defined likelihood term; gradients are clean and parallelizable (section 05's additive structure over positions — a sequence's terms don't depend on sampling).
- **The gap it creates:** at inference the model feeds on *its own* outputs; a model that has only seen perfect prefixes can drift when it makes an error (its conditional was never trained on that history). Consequences and remedies — scheduled sampling, sequence-level objectives, RL fine-tuning (beyond scope; stated honestly as the known train/generation mismatch).
- **What stays true either way:** the *mathematics* of the objective is MLE of the causal factorization; the exposure problem is about distribution shift between training contexts and generation contexts, not an error in the derivation.

## Honest Details

- **Label smoothing:** soften the target from one-hot to $(1-\epsilon)$ on the true token, $\epsilon/V$ elsewhere — trades a little MLE faithfulness for less overconfidence (section 06: the modified target is a different $p$; the model now fits *that* distribution). Widely used in machine translation; less in modern LLM recipes — an engineering dial, not new math.
- **Tokens, not words:** the loss is per *token*; tokenization granularity changes $T$ and every per-token number (chapter 03's comparability warning).
- **Context truncation:** if sequences exceed the window, only the last $t$ tokens contribute terms — the likelihood of what fits; long-document modeling is partly about how much of the product you actually evaluate.
- **Multiple sequences** in a batch: average the per-sequence means (or pool all tokens — consistent denominators matter for comparing runs).

## Connection to AI

- **This formula is the loss printed by every training run** — "loss: 3.2" in logs is $\mathcal{L}$ in nats per token, nothing else.
- **Alignment stages start from here:** preference optimization and RL methods modify or constrain this objective (section 09's orbit, section 10's KL-regularized variants) — but the supervised pretraining loss *is* this chapter.
- **Section 06's chain rule for entropy** ($H(X_1,\ldots,X_T) = \sum H(X_t \mid X_{<t})$) and this section's probability chain rule are the same identity under expectation — the objective estimates the entropy rate of the corpus from data (the deepest way to say what pretraining does).
- **Curriculum consistency check:** linear/logistic regression's loss (section 07) was cross-entropy at one position; an LLM applies the identical loss at $T-1$ positions and averages — no new loss appears anywhere in the course.
- **Evaluation losses** on held-out data are this formula on unseen sequences — section 04's held-out discipline: report *that* number, not training loss (section 08's warning).

## Common Mistakes

- Averaging over the wrong count (dividing by batch size when you summed over tokens, or vice versa) — silently rescales gradients and makes losses incomparable across runs.
- Forgetting the **causal** conditioning during debugging — if loss drops suspiciously fast, check the mask (leakage makes $\mathcal{L}$ artificially low — section 04's leakage lesson).
- Computing $-\ln P$ of the *whole generated sequence* instead of the true tokens — teacher forcing evaluates data likelihood, not free-running quality (different metric).
- Mixing nats and bits across runs (factor $\ln 2$) — logging conventions matter for comparisons.
- Believing lower training $\mathcal{L}$ means a better chat model — capability, safety, and usefulness are not the likelihood of the training corpus (honest: this objective teaches next-token prediction, full stop).
- Ignoring padding/masked positions in the average — padded tokens must not enter the sum or the denominator.

## Exercises

1. Factor $P(w_1, w_2, w_3)$ and write the sequence NLL as a sum; then write the per-token mean.
2. Compute $\mathcal{L}$ (per-token mean) for: $P(w_2 \mid w_1) = 0.5$, $P(w_3 \mid w_1, w_2) = 0.5$, sequence length 3.
3. Same setup with probabilities $0.9$ and $0.9$: compute; which corpus is the model "more surprised" by, and by how much in nats/token?
4. Connect to section 06: the target distribution at position $t$ is one-hot — write its cross-entropy with the model's $P(\cdot \mid w_{\le t})$ and confirm it equals $-\ln P(w_{t+1} \mid w_{\le t})$.
5. A batch of 8 sequences, each length 100 (99 prediction terms each): how many log-probability terms enter the mean if we average over all tokens in the batch?
6. Teacher forcing in one sentence: what is conditioned on during training, and what is conditioned on during generation?
