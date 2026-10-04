# Attention: Scaled Dot-Product Attention

## Why It Matters for AI

**Attention** is the operation that made modern language models possible: every token looks at every other token and *mixes* information according to learned relevance. The formula is one line of linear algebra plus a softmax — but each piece has a reason: what queries, keys, and values are; why the dot product is scaled by $1/\sqrt{d_k}$ (derived below, not folklore); and how causal masking turns an all-to-all operation into an autoregressive one.

## Queries, Keys, Values

Each token vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ (chapter 01: a row of $X$) is projected three ways by **learned** matrices (section 01's linear maps — rows are parameters, trained by backprop):

$$
\mathbf{q} = W_Q \mathbf{x}, \qquad
\mathbf{k} = W_K \mathbf{x}, \qquad
\mathbf{v} = W_V \mathbf{x}
\qquad
W_Q, W_K \in \mathbb{R}^{d_k \times d_{\text{model}}},\; W_V \in \mathbb{R}^{d_v \times d_{\text{model}}}
$$

Database intuition (the names are literal): a token *asks* with $\mathbf{q}$ ("what am I looking for?"), *announces* with $\mathbf{k}$ ("what I contain"), and *offers* $\mathbf{v}$ ("what I'll contribute if selected"). Relevance of token $i$ to token $j$ = how well $j$'s key matches $i$'s query:

$$
e_{ij} = \mathbf{q}_i^\top \mathbf{k}_j \qquad (\text{section 01's dot product: magnitude} \times \cos\theta)
$$

For the whole sequence (rows = tokens, chapter notation):

$$
E = Q K^\top \in \mathbb{R}^{n \times n}
$$

— the **all-pairs score matrix**: entry $(i, j)$ = compatibility of query $i$ with key $j$.

## Softmax: Scores Become Weights

Scores are unbounded reals; attention needs weights that form a distribution (section 03: probabilities; section 06: normalized positive scores). **Row-wise softmax:**

$$
A_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})}
\qquad\Longrightarrow\qquad
A_{ij} \ge 0,\quad \sum_j A_{ij} = 1 \ \ \text{(each row is a probability vector)}
$$

Row $i$ reads as a **conditional distribution over positions given token $i$'s query** — exactly the kind of object sections 03–06 spent four chapters on. The exponential: positive (any score valid), amplifies larger scores (preference for better matches), differentiable everywhere (section 02's smoothness); and softmax is invariant to adding a constant $c$ to an entire row ($e^{x+c} / \sum e^{x'+c} = e^{x}/\sum e^{x'}$ — only relative scores matter).

**Output:**

$$
\boxed{\;\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V\;}
$$

Each output row $\mathbf{o}_i = \sum_j A_{ij} \mathbf{v}_j$ — a **convex combination** (weights ≥ 0, sum 1) of value vectors: token $i$'s new representation is the weighted average of everyone it attended to. Matrix view: $A$ (stochastic) times $V$ — linear algebra all the way down.

## The $1/\sqrt{d_k}$ Derivation

**Question.** Why divide the scores by $\sqrt{d_k}$?

**Setup.** Before training settles, model the projections as roughly independent, zero-mean, unit-variance components: $q_i, k_j \overset{\text{iid}}{\sim}$ mean $0$, variance $1$ (the standard initialization-scale regime from section 04's variance bookkeeping). Then a raw dot product of length $d_k$:

$$
\mathbf{q}^\top \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i
$$

Each product $q_i k_i$: mean $\mathbb{E}[q_i]\mathbb{E}[k_i] = 0$, variance $\mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$ (independence). Sum of $d_k$ terms:

$$
\operatorname{Var}(\mathbf{q}^\top \mathbf{k}) = d_k
\qquad\Longrightarrow\qquad
\text{std} = \sqrt{d_k}
$$

**The problem.** Softmax *saturates* when its inputs spread out (chapter 01 of section 08's lesson in new clothes). With score standard deviation $\sqrt{d_k}$:

| $d_k$ | score std | softmax behavior at typical spread |
|---|---|---|
| 2 | $1.4$ | gentle — weights like $0.67/0.33$ |
| 64 | $8$ | dominant entry $\sim e^{8} \approx 3000\times$ the rest — near one-hot |
| 512 | $22.6$ | effectively hard selection — gradient $\approx 0$ |

Near-one-hot attention: (a) gradients through softmax vanish (the network can't learn softer mixes), (b) the operation degenerates into hard lookup (the reason the original paper's ablations show it matters). This is exactly chapter 01's sigmoid saturation — large inputs, flat response, dead learning.

**The fix.** Divide scores by $\sqrt{d_k}$:

$$
\operatorname{Var}\!\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

Score spread becomes $O(1)$ *regardless of head width* — softmax operates in its responsive regime, every head (whatever $d_k$) sees comparable input scales, and learning can decide sharpness — not the dimensionality. (Equivalent reading: $\sqrt{d_k}$ is the temperature knob — section 10 revisits temperature: dividing by it is fixing the effective temperature so it stops drifting with $d_k$.)

**Worked illustration** ($d_k = 2$, raw scores $(1, 0)$ — as if $\mathbf{q}^\top\mathbf{k} = 1$ and $0$):

- **Unscaled:** $\operatorname{softmax}(1, 0) = \left(\frac{e}{1 + e}, \frac{1}{1+e}\right) = (0.731,\ 0.269)$
- **Scaled** (divide by $\sqrt2 \approx 1.414$): $\operatorname{softmax}(0.707, 0) = (0.670,\ 0.330)$

The scaled version stays softer; the gap widens as $d_k$ grows (the table above) — one line, entire effect.

## Causal Masking: Autoregressive Attention

**Problem.** In a language model, token $i$'s output must depend only on tokens $1..i$ — but $E = QK^\top$ lets every token see every other, including the future. Leakage: the model could cheat by reading the answer.

**Fix.** Add $-\infty$ to forbidden scores **before** the softmax:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}, \qquad
A = \operatorname{softmax}\!\big((QK^\top + M)/\sqrt{d_k}\big)
$$

$\exp(-\infty) = 0$: every future position gets **exactly zero** attention weight; the row's softmax then renormalizes over $j \le i$ alone. Row $i$ becomes a probability distribution over the *prefix* $1..i$ — precisely the conditional structure of section 04's chain rule:

$$
H(X_1, \ldots, X_T) = \sum_t H(X_t \mid X_{<t})
$$

— the factorization that makes autoregressive modeling legitimate (section 10 trains each row toward the true next-token conditional). Causality is not an architectural extra; it's a $-\infty$ in the right triangle.

## Worked Micro-Example

$n = 2$, $d_k = 2$, identity projections ($W_Q = W_K = I$) so $Q = K = I_2$, and $V = I_2$ (values = the token vectors themselves):

$$
\frac{QK^\top}{\sqrt2} = \frac{I}{\sqrt2} = \begin{bmatrix} 0.707 & 0 \\ 0 & 0.707 \end{bmatrix}
$$

Row-wise softmax (each row: $(0.707, 0) \to (0.670, 0.330)$):

$$
A = \begin{bmatrix} 0.670 & 0.330 \\ 0.330 & 0.670 \end{bmatrix},\qquad
O = A V = A
$$

Each token keeps $67\%$ of itself and takes $33\%$ of the other — attention as *interpolation*. With the causal mask ($M_{12} = -\infty$): row 1 becomes $(1, 0)$ — token 1 attends only to itself; row 2 unchanged $(0.330, 0.670)$ — future blocked, past allowed. (Add masking as an exercise; the renormalization is the point.)

```python
import math
def softmax(row):
    m = max(row); e = [math.exp(x - m) for x in row]
    s = sum(e); return [xi / s for xi in e]
print(softmax([1/math.sqrt(2), 0]))   # [0.670, 0.330]
```

## Complexity (Why Context Costs Quadratically)

$QK^\top$ computes all $n^2$ pairs: time and memory grow as $O(n^2 d)$ — every added token attends to, and is attended by, all previous ones. This quadratic term is the mathematical root of context-length engineering (sparse/linear attention variants attack exactly this matrix's cost). Honest framing: the *formula* is simple; its pairwise enumeration sets the economics of long contexts.

## Connection to AI

- **$W_Q, W_K, W_V$ are ordinary trained parameters** — backprop (section 08) differentiates through softmax and matmuls; nothing about attention resists gradients (chapter 03 of section 08 applies unchanged).
- **Per-token weighted mixing = conditional computation:** unlike a fixed convolution, attention *data-dependently* routes information — token $i$'s neighborhood is decided by its query (the same content-addressing idea as retrieval, now soft).
- **Softmax row-stochasticity** gives attention maps a probabilistic reading (each row sums to 1) — visualization and analysis tools treat them as distributions, section 06's language.
- **Section 10 assembles these rows** into next-token predictions: attention output → output projection → vocabulary logits → softmax → cross-entropy.
- **The scaling law of pairs:** the $n^2$ matrix is also why training/inference cost attention is *the* item on the compute budget — architecture search ≈ finding cheaper surrogates for this one product.

## Common Mistakes

- Forgetting the row-wise softmax (normalizing the whole matrix instead of each row) — weights must sum to 1 **per query token**.
- Applying the mask *after* softmax — too late; masked entries must be $-\infty$ *before* exponentiation, then renormalized.
- Dropping $1/\sqrt{d_k}$ — training may still work at small $d_k$, then fails mysteriously at larger widths (the derivation's variance grows silently).
- Confusing $\sqrt{d_k}$ with $d_k$ — variance of the sum is $d_k$, so the *standard deviation* to divide by is $\sqrt{d_k}$.
- Using $V$ for both vocabulary size and value matrix (common notation collision — context distinguishes; this repo keeps $V$ = values, $V$ the vocab only in embedding chapters).
- Assuming attention weights are meaningful explanations (a common misreading: they are routing coefficients, not causal attribution).

## Exercises

1. Compute row-wise softmax by hand: $\operatorname{softmax}(2, 1)$ (to 3 decimals). Then softmax of $(2/\sqrt2, 1/\sqrt2)$ — which is sharper, and why does scaling matter here?
2. Show the variance claim for $d_k = 9$: if $\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = 9$, what is the std of raw scores and of scaled scores?
3. Shapes: $X \in \mathbb{R}^{n \times 512}$, $d_k = 64$ per head. Give shapes of $Q$, $QK^\top$, $A$, $AV$ (single head, $d_v = 64$).
4. Causal mask: with scores $\begin{bmatrix}1 & 0.5\\ 0.5 & 1\end{bmatrix}$ and $M_{12} = -\infty$, compute row 1 of $A$ after masking (soft $\max$ of $(1, -\infty)$).
5. Why must $W_Q$ and $W_K$ be *separate* matrices? Give a one-sentence argument from what queries and keys must do differently.
6. Interpret $O = AV$ in one sentence as a weighted average — identify the weights and what is being averaged.
