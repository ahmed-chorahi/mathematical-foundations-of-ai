# The Pipeline: From Tokens to Next-Token Probabilities

## Why It Matters for AI

"What does the model actually compute?" — one function, end to end: a prefix of token IDs in, a probability distribution over the next token out. This chapter walks that function stage by stage with shapes and one worked numerical example, naming each stage's mathematics (all already met: embeddings and blocks from section 09, logits and softmax from section 07, conditionals from section 03).

## The Forward Pipeline

Given prefix $(w_1, \ldots, w_t)$:

**1. Embed + position (section 09, chapter 01).**

$$
X_0 \in \mathbb{R}^{t \times d_{\text{model}}}
\quad\text{with row } i = E_{w_i,:} + \mathbf{P}_i
$$

**2. Transformer stack (section 09, chapter 03).** $N$ causal blocks:

$$
X_N = \text{Block}_N(\cdots \text{Block}_1(X_0)) \in \mathbb{R}^{t \times d_{\text{model}}}
$$

Every attention line carries the $-\infty$ causal mask — position $i$'s row only mixed $1..i$ (chapter 02 of section 09). The result: row $t$ is a **context summary** of the prefix alone, $\mathbf{h}_t = (X_N)_{t,:}$ — precisely a representation of $w_{\le t}$.

**3. Unembedding: hidden state → logits.** Project to vocabulary coordinates with a learned matrix $W_U \in \mathbb{R}^{V \times d_{\text{model}}}$ (often *tied* to the embedding matrix $E$ from section 09 — same $V \times d$ matrix used forward and backward, halving that parameter block):

$$
\boldsymbol{\ell}_t = W_U \mathbf{h}_t + \mathbf{b}_U \in \mathbb{R}^{V}
$$

$\ell_{t,j}$ = **logit** for vocabulary item $j$ — an unnormalized score (section 07's logits, now vocabulary-sized). *Every one of these scores is a linear readout of a contextualized representation — no probability yet.*

**4. Softmax: logits → distribution (section 07).**

$$
P(w_{t+1} = j \mid w_{\le t}) = \frac{e^{\ell_{t,j}}}{\sum_{j'=1}^{V} e^{\ell_{t,j'}}}
$$

A probability vector: non-negative, sums to 1 — **the model's conditional distribution over the next token, given the prefix**. This is the object section 03 spent a chapter on ($P(\cdot \mid X{=}x)$, now with $x$ = the prefix), and the object the causal mask guaranteed would only depend on the prefix.

**Reading the loop:** generation appends the sampled/argmax token, re-runs (incrementally) the pipeline, repeats — one distribution per step.

## Worked Micro-Example

Prefix probabilities for a 3-word vocabulary at some position — logits from the unembedding layer:

$$
\boldsymbol{\ell} = (2.0,\ 1.0,\ 0.1) \quad \text{for} \quad (\texttt{cat},\ \texttt{dog},\ \texttt{the})
$$

Softmax (chapter of section 07 — hand-checkable):

$$
e^{2.0} = 7.389,\quad e^{1.0} = 2.718,\quad e^{0.1} = 1.105;\qquad \text{sum} = 11.212
$$

$$
P = (0.659,\ 0.242,\ 0.099)
$$

The model assigns the next token `cat` probability $0.659$, `dog` $0.242$, `the` $0.099$ (verified: sums to $1.000$ ✓). **Claim about the future:** if the true next token is `cat`, the training loss at this position is $-\ln 0.659 = 0.417$ nats (chapter 02 computes the sequence version).

```python
import math
def softmax(l):
    e = [math.exp(x) for x in l]; s = sum(e)
    return [xi / s for xi in e]
print(softmax([2.0, 1.0, 0.1]))   # [0.659, 0.242, 0.099]
```

## What Each Stage Contributes

| stage | mathematical object | from |
|---|---|---|
| embedding lookup | row of $E$: discrete $\to$ $\mathbb{R}^{d}$ | section 09 |
| position add | bounded offset breaking permutation symmetry | section 09 |
| causal attention | masked softmax mixing prefix rows | section 09 |
| stack of blocks | deep composition (mix ↔ transform) | section 09 |
| unembedding $W_U\mathbf{h}$ | linear scores in $\mathbb{R}^V$ | section 01 |
| softmax | scores $\to$ conditional distribution | sections 03, 07 |
| **output** | $P(w_{t+1} \mid w_{\le t})$ | section 03 |

Two final facts worth stating plainly:

- **The network is a conditional model.** Nothing in the pipeline breaks the causal structure: the mask + prefix-only re-run per step means the distribution *is* $P(\cdot \mid \text{prefix})$ by construction — the same conditional whose entropy $H(W_{t+1} \mid W_{\le t})$ measures the task's difficulty (section 04).
- **One matrix between representation and vocabulary:** $W_U$ turns $d_{\text{model}}$ numbers into $V$ numbers — the linear-readout shape of section 07's classifier head, applied at every position.

## Connection to AI

- **Chat models, code models, translation systems** all terminate in this pipeline — differences live in the data, size, and alignment stages *after* the mathematics, not in a different formula.
- **Retrieval, tools, multi-modal inputs** enter by expanding what constitutes the prefix (more token rows) — the conditional and its training objective are unchanged.
- **Vector databases / embeddings search** are the *embedding* half of section 09 spun out as a product — same $E$-style matrices, different downstream use.
- **Efficiency engineering** (KV caching, quantization) compresses the pipeline's *execution* while preserving the function $P(w_{t+1} \mid w_{\le t})$ — correctness is defined by this chapter's math.
- **Evaluation starts here:** log-probabilities from step 4 are the raw material for loss (chapter 02), perplexity (chapter 03), and calibration analysis (section 06).

## Common Mistakes

- Confusing **logits with probabilities** — $\ell$ is unbounded; only post-softmax values are usable as $P$ (feeding logits into a log-loss directly is a category error).
- Expecting the distribution to depend on the *future* — it cannot (mask + inference loop); a model "peeking" means a broken mask (chapter 02's leakage warning).
- Forgetting that $V$-sized softmax is over **this tokenizer's** vocabulary — probabilities are over token IDs, not words or bytes; tokenization changes the space (matters for perplexity comparisons — chapter 03).
- Assuming position $t$'s logits only depend on $w_t$ — they depend on the *entire prefix* (that's the point); each row of $X_N$ is contextually mixed.
- Mixing up axis order: $W_U \mathbf{h}$ with $\mathbf{h}$ as column vs row-convention $ \mathbf{h} W_U^\top$ — consistent with section 01's transpose discipline.
- Treating the top logit's magnitude as a probability — it is a score; only differences *after* softmax exponentiation carry probabilistic meaning.

## Exercises

1. Shapes: $d_{\text{model}} = 2048$, $V = 50{,}000$, prefix length $t = 17$. Give shapes of $X_0$, $\mathbf{h}_{17}$, $\boldsymbol{\ell}_{17}$, $P_{17}$, and the parameter count of $W_U$.
2. Compute softmax by hand: logits $(0, 0)$ over a 2-token vocab — why must every position's distribution give at least $1/V$ somewhere?
3. Logits $(3, 0)$: compute $P$ (2 decimals). What is $-\ln P(\text{top token})$?
4. Why does the causal mask make the output *legitimately* a conditional $P(w_{t+1} \mid w_{\le t})$? (One sentence referencing section 09's mask.)
5. Untied vs tied $W_U$: what are the parameter counts for $V = 50{,}000$, $d = 2048$ if $W_U$ is separate from $E$? If tied?
6. A training batch has sequences of length 512 — at how many positions does this pipeline produce distributions (and hence losses)? (Chapter 02 uses the answer.)
