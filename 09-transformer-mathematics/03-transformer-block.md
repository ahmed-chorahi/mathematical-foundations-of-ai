# The Transformer Block: Multi-Head Attention, Residuals, Layer Norm, Feed-Forward

## Why It Matters for AI

One attention operation is a toy; the *block* — multi-head attention, residual connections, layer normalization, and a feed-forward network — is the unit that stacks dozens or hundreds of times into a language model. Every component is a formula from earlier sections wearing a new name: outer products, z-scores, gradient highways, and a two-layer network. This chapter assembles the block and counts what it costs.

## Multi-Head Attention

**Problem with one attention:** a single $QK^\top$ uses one projection subspace — one kind of similarity at a time. Syntax, coreference, positional patterns, rare-token signals compete for the same score function.

**Fix:** $h$ **heads**, each with its own $W_Q^{(j)}, W_K^{(j)}, W_V^{(j)}$ acting on a slice of the width ($d_k = d_v = d_{\text{model}}/h$):

$$
\text{head}_j = \operatorname{Attention}\!\big(X W_Q^{(j)},\, X W_K^{(j)},\, X W_V^{(j)}\big) \in \mathbb{R}^{n \times d_v}
$$

$$
\boxed{\;\text{MultiHead}(X) = \big[\text{head}_1 \| \cdots \| \text{head}_h\big]\, W_O
\qquad W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}\;}
$$

- $[\;\|\;]$ = concatenation along features (section 01: block matrices — the concatenation of $n \times d_v$ blocks is $n \times hd_v$); $W_O$ mixes the heads' outputs back into $d_{\text{model}}$.
- **Why separate projections beat one big one:** different heads specialize — some attend locally, some to syntactic heads, some to rare tokens (empirical findings from probing studies — honest as *observations*, not theorems). The concatenation + $W_O$ lets each head's output flow to where it's needed; with a single head, all relations must share one scoring function.
- **Worked shapes:** $d_{\text{model}} = 512$, $h = 8$ ⇒ $d_k = d_v = 64$; $W_Q^{(j)} \in \mathbb{R}^{64 \times 512}$ per head ($8 \times 64 \times 512 = 262{,}144 = 512 \times 512$ total — splitting the width costs exactly the same parameters as one full-width head set; the gain is *independent subspaces*, not parameter count); $W_O \in \mathbb{R}^{512 \times 512}$.
- The per-head $\sqrt{d_k}$ scaling (chapter 02) uses **each head's** $d_k = 64$ here — not $d_{\text{model}}$.

## Residual Connections

$$
\mathbf{x}^{(\ell+1)} = \mathbf{x}^{(\ell)} + F(\mathbf{x}^{(\ell)})
$$

$F$ = any sub-layer (attention, FFN). From section 08's gradient analysis:

$$
\frac{\partial \mathbf{x}^{(\ell+1)}}{\partial \mathbf{x}^{(\ell)}} = I + \frac{\partial F}{\partial \mathbf{x}^{(\ell)}}
$$

The identity term means gradients contain an **additive path that never decays** — even if $\partial F/\partial x$ is small or vanishing, the $I$ carries signal at gain exactly $1$ across any depth (chapter 04 of section 08 named this cure; here it is installed). Residuals are why 96-layer stacks train while 20-layer plain sigmoid stacks did not.

## Layer Normalization

Per token vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ (across its features — *not* across the batch), compute mean and variance **exactly as section 04's standardization**:

$$
\mu = \frac{1}{d}\sum_i x_i, \qquad
\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2, \qquad
\operatorname{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
$$

- $\gamma, \beta \in \mathbb{R}^{d}$: **trainable** scale and shift (let the network undo or reshape normalization if useful — affine recovery; $z$-score with learned parameters).
- $\varepsilon$: prevents division by $0$ for constant vectors (numerical hygiene, section 04's division caveats).
- **Why per token:** activations drift layer by layer (section 04: means/variances shift); re-standardizing *each* vector keeps every layer's input in a healthy range → gradients stay $O(1)$ through depth (section 08's gradient-health requirement), and the $\sigma^2 + \varepsilon$ mirrors statistics' care about variance denominators.
- **Placement in the block (post-norm, original):** $\mathbf{x} \leftarrow \mathbf{x} + \operatorname{LN}(F(\mathbf{x}))$ variants exist; both orderings (pre-LN/post-LN) differ in training stability — an engineering fact, notation kept simple here.

## The Feed-Forward Block

$$
\operatorname{FFN}(\mathbf{x}) = W_2\, \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

- Exactly a one-hidden-layer network from section 08 — applied **to each token independently** (same weights for every position: a position-wise MLP).
- Width: $d_{\text{ff}} \approx 4 \cdot d_{\text{model}}$ (512 → 2048 → 512 in the original) — the expansion gives room for the nonlinearity to carve per-token transformations; then compress back.
- **Role:** attention *mixes across positions*; FFN *processes each position* — alternating them is the block's division of labor (mix, then transform, then mix, ...).
- Activation: ReLU family / GELU (section 08's landscape).

## Assembling the Block

$$
\begin{aligned}
\mathbf{H} &= \mathbf{X} + \operatorname{MultiHead}(\mathbf{X}) && \text{(mix + residual)}\\
\mathbf{H}' &= \operatorname{LayerNorm}(\mathbf{H}) && \text{(re-standardize)}\\
\mathbf{Y} &= \mathbf{H}' + \operatorname{FFN}(\mathbf{H}') && \text{(transform + residual)}\\
\mathbf{Z} &= \operatorname{LayerNorm}(\mathbf{Y}) && \text{(re-standardize)}
\end{aligned}
$$

Stack $N$ of these ($\mathbf{X}^{(\ell+1)} = \text{Block}(\mathbf{X}^{(\ell)})$); causal mask lives inside the attention line (chapter 02).

**Parameter count per block** ($d = 512$, $h = 8$, $d_{\text{ff}} = 2048$, counting matrices, ignoring small biases):

| component | parameters |
|---|---|
| $W_Q, W_K, W_V, W_O$ (each $512 \times 512$) | $4 \times 262{,}144 = 1{,}048{,}576$ |
| FFN: $W_1\ (2048 \times 512)$ + $W_2\ (512 \times 2048)$ | $2 \times 1{,}048{,}576 = 2{,}097{,}152$ |
| LayerNorm: 2 norms × ($\gamma, \beta$ of size 512) | $2{,}048$ |
| **total** | $\approx \mathbf{3.15}$ **million / block** |

$N$ blocks ≈ $3.15N$ M (+ embeddings and output head, chapter 01 / section 10) — arithmetic you can do before writing any code (the skill behind model-size claims).

**Forward pass, end to end:** embed + position (ch 01) → $[\text{Block}]^N$ (this chapter) → per-position projection to vocabulary logits (section 10) → softmax + cross-entropy (sections 06–07) → loss → backprop (section 08) through *all* of it.

## Connection to AI

- **Every GPT/LLaMA-style model is this block repeated** with small variations (different FFN activation, RMSNorm instead of LayerNorm, rotary positions inside attention) — the mathematics of this chapter *is* the model class.
- **Depth = compositional reuse:** each block can condition the next on freshly mixed information — deeper networks express longer dependency chains (the empirical reason depth beats width for language).
- **Compute scaling:** $3.15$M/block × depth × context length arithmetic sets training budgets — the block is the unit of cost accounting.
- **Training machinery unchanged:** no new loss, no new optimizer — cross-entropy + SGD-family + backprop from sections 05–08 flows through softmax masks, LayerNorms, and residuals (LayerNorm's $\sqrt{\sigma^2 + \varepsilon}$ is just another local derivative node).
- **Why ablations remove one component at a time:** each line above is independently testable — residuals, norms, heads, FFN width all have published removal studies; the block's design is empirical engineering on top of exact mathematics.

## Common Mistakes

- Normalizing across the batch or across positions instead of across the token's features — wrong axes; LayerNorm is per-row, per-token.
- Concatenating heads along the sequence axis — heads concatenate along **features** ($[\;\|]$ in $n \times hd_v$).
- Using $d_{\text{model}}$ instead of $d_k$ in the $\sqrt{d_k}$ scale with multiple heads — off by $\sqrt{h}$; sharpness drifts.
- Forgetting $W_O$ after concatenation — head outputs never interact; the block degrades to parallel independent mixers.
- Residual addition with mismatched shapes (adding before/after a projection that changed width) — the residual branch must restore $d_{\text{model}}$.
- Counting parameters without the $4\times$ FFN factor — FFNs dominate block size (2.1M of the 3.15M above).
- Applying LayerNorm to the sequence axis and wondering why training diverges — axis errors here produce healthy-looking code with dead gradients.

## Exercises

1. Shapes: $h = 12$, $d_{\text{model}} = 768$ — find $d_k$ per head and the total size (parameters) of $W_Q, W_K, W_V, W_O$ combined.
2. Compute the FFN parameter count for $d_{\text{model}} = 768$, $d_{\text{ff}} = 3072$ (two matrices, no biases).
3. LayerNorm by hand: $\mathbf{x} = (1, 3, 5)$, $\gamma = 1$, $\beta = 0$, $\varepsilon = 0$ — compute $\mu$, $\sigma^2$, and the output. (What familiar vector is this?)
4. Residual gradient: if $\partial F/\partial x = 0.01 I$, what is the per-block gradient factor with and without the residual? What happens over 50 layers (give both numbers)?
5. Why do attention and FFN alternate instead of attention-only? One sentence about the division of labor.
6. Model arithmetic: a 32-block model with $d = 1024$, $d_{\text{ff}} = 4096$, $h = 16$ — compute block parameters (4 attention matrices + 2 FFN matrices) and the total for 32 blocks.
