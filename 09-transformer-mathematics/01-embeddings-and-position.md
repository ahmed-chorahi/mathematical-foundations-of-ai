# Embeddings and Position Encodings

## Why It Matters for AI

A transformer processes *vectors*, not words: every token becomes a point in $\mathbb{R}^{d_{\text{model}}}$ before anything else happens. Two questions have exact mathematical answers: how does a discrete token become a vector (**embedding**), and how does the model know the *order* of tokens when attention treats inputs as a set (**position encoding**)? Both are small formulas with large consequences.

## Token Embeddings: A Matrix Row

Vocabulary of size $V$: tokens $\{t_1, \ldots, t_V\}$. An embedding is a learned matrix

$$
E \in \mathbb{R}^{V \times d_{\text{model}}}
$$

Token $t_i$'s vector = **row $i$ of $E$**:

$$
\mathbf{e}_i = E_{i,:} \in \mathbb{R}^{d_{\text{model}}}
$$

Equivalent view (section 07's one-hot machinery): one-hot $\mathbf{u}_i \in \mathbb{R}^V$, then $\mathbf{e}_i = E^\top \mathbf{u}_i$ — a matrix–vector product that *selects* a row. The lookup formulation is what code uses (no multiply needed; just index).

- $E$ is **trained** like every other parameter (backprop sends gradient $\partial L / \partial E_{i,:}$ only to rows that appeared — sparse updates, section 03's one-hot sparsity showing up computationally).
- Parameter cost: $V \cdot d_{\text{model}}$ — often one of the largest single matrices in the model. Scale arithmetic: $V = 50{,}257$, $d_{\text{model}} = 12{,}288$ gives $50{,}257 \times 12{,}288 \approx 617.6$ million parameters in the embedding alone (verified: $50{,}257 \times 12{,}288 = 617{,}558{,}016$).
- Tokens with similar *usage* learn similar rows — distributional structure (section 06's MI intuition: rows organize by what predicts what) emerges from training, not from the formula.

A sequence $t_1, \ldots, t_n$ becomes the matrix

$$
X = \begin{bmatrix} \mathbf{e}_{t_1} \\ \vdots \\ \mathbf{e}_{t_n} \end{bmatrix} \in \mathbb{R}^{n \times d_{\text{model}}}
$$

## The Missing Ingredient: Order

**Problem.** Attention (chapter 02) computes interactions between *rows* of $X$ symmetrically — permute the rows and the set of pairwise interactions is unchanged. As a function of a *set*, a stack of attention layers cannot distinguish "dog bites man" from "man bites dog": both contain the same vectors. Formally: without position information the network is **permutation-equivariant** (output rows permute with input rows) — sequence structure is invisible.

**Fix.** Add a **position encoding** $P \in \mathbb{R}^{n \times d_{\text{model}}}$ to the embeddings:

$$
X_0 = X + P
$$

(each row gets the encoding of its position; addition, not concatenation — keeps $d_{\text{model}}$ fixed through the stack).

### Sinusoidal Encodings (The Original Design)

$$
P_{\text{pos},\, 2i} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right),\qquad
P_{\text{pos},\, 2i+1} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

Reading the formula (dimension $i$ gets a wave of wavelength $\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}$):

- **Low coordinates ($i$ small):** short wavelengths — position changes rapidly; fine-grained position signal.
- **High coordinates ($i$ large):** long wavelengths (up to $10000$-scale periods) — coarse, slowly varying position signal.
- **Geometric ladder of wavelengths:** every dimension pair is a ruler at a different scale — together, $(\sin, \cos)$ pairs across dimensions pin down position the way a mixed-radix number system does (bounded, unique over long ranges).
- **Relative-position structure:** angle-addition gives $\text{PE}(\text{pos}+k) = $ a *fixed linear rotation* of $\text{PE}(\text{pos})$ (same rotation matrix for every pos, determined by $k$) — so relative offsets act as consistent transformations the network can learn to exploit (stated as the design property; the proof is the trig identity $\sin(a+b) = \sin a\cos b + \cos a \sin b$ in matrix form).
- **Bounded:** every entry in $[-1, 1]$ — comparable scale to embeddings (they get *added*; wildly different scales would drown one signal in the other — section 04's scaling hygiene again).

*Worked values:* $\text{pos} = 1$, coordinate $i = 0$ ($10000^0 = 1$): $\sin(1) = 0.8415$, $\cos(1) = 0.5403$ — the first coordinate pair at position 1. (At pos $= 0$: $\sin(0) = 0$, $\cos(0) = 1$ — a canonical "start" pattern.)

**Alternatives (honest landscape):** *learned* position embeddings (another $n \times d$ trainable table — simple, used in GPT-1-era and several modern models); *relative* schemes (T5's relative bias, ALiBi) and rotary embeddings (RoPE, applied inside attention) — the original sinusoids are not the last word, but every scheme solves the same set-to-sequence gap with positional information of bounded scale.

## Connection to AI

- **Everything downstream is vector algebra:** after this chapter, a language model never sees a token ID again — only $X_0$ flowing through chapter 03's blocks.
- **$V \times d$ embeddings dominate parameter budgets** in small models and remain large in big ones — often shared (tied) with the output layer to save parameters (section 10's pipeline uses the same matrix in reverse).
- **Position encoding choices explain context behaviors:** models must *learn* position from these added signals; how well they extrapolate to longer sequences than training is tied to the encoding scheme — an active empirical question, precisely because the formula determines what "position" means to the network.
- **Permutation-equivariance without positions** is also *why* the same architecture works for sets (point clouds, molecules) when position signals are deliberately omitted — the failure case above becomes a feature.
- **Scale discipline:** $X + P$ assumes comparable magnitudes — the embedding/position scale mismatch is a real debugging failure (washed-out positions).

## Common Mistakes

- Forgetting positions entirely for sequential tasks — the model literally cannot tell order (it can exploit token *co-occurrence* patterns, which leaks some order, but the sequence structure is not represented — a subtle failure mode, not an excuse).
- Off-by-one positions (0-index vs 1-index) — shifts every learned relative pattern by one; consistency matters, the convention doesn't.
- Rescaling embeddings but not positions (or vice versa) — one signal drowns the other in the sum.
- Assuming sinusoids *give* grammar — they supply a coordinate system; the transformer learns how to use it.
- Treating rows of $X$ as words — they are *context-ready vectors*; the first attention layer will mix them (chapter 02) before any output is produced.
- Counting position parameters as trainable for sinusoids — they are **fixed** (no gradients); learned tables differ.

## Exercises

1. Token $t_5$, $V = 10{,}000$, $d_{\text{model}} = 8$: what are the shapes of $E$ and $\mathbf{e}_5$? Express the lookup as a one-hot product.
2. Compute the embedding-table size in parameters for $V = 32{,}000$, $d_{\text{model}} = 768$.
3. Compute $P_{\text{pos}=1,\,i=0}$ and $P_{\text{pos}=\pi/2,\,i=0}$ (both sin and cos; note the $\cos$ value at $\pi/2$).
4. Why does adding *fixed* sinusoidal vectors make order visible to attention? (One sentence: what breaks the permutation symmetry.)
5. At $i = d_{\text{model}}/2$, the wavelength is $2\pi \cdot 10000$ — explain in one line why having both very short and very long wavelengths matters for encoding long sequences.
6. Two sequences are the same bag of tokens in different orders. Without position encodings, write down why a self-attention layer's outputs for both are equal up to row permutation. (Reference the pairwise score matrix.)
