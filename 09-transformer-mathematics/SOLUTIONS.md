# 09 — Transformer Mathematics: Solutions

## 01-embeddings-and-position.md

**1. Shapes and one-hot product.** $E \in \mathbb{R}^{10{,}000 \times 8}$, $\mathbf{e}_5 \in \mathbb{R}^{8}$ (row 5 of $E$). As a product: $\mathbf{e}_5 = E^\top \mathbf{u}_5$ with $\mathbf{u}_5 \in \mathbb{R}^{10{,}000}$ the one-hot vector $(0,\ldots,1,\ldots,0)$ — the matrix–vector form of a row lookup (section 07's one-hot machinery).

**2. Table size.** $32{,}000 \times 768 = 24{,}576{,}000 \approx \mathbf{24.6}$ **million parameters** — one matrix, often the model's single largest.

**3. Sinusoidal values.** $i = 0 \Rightarrow 10000^{0} = 1$: pos $= 1$: $\sin(1) = 0.8415$, $\cos(1) = 0.5403$. Pos $= \pi/2$: $\sin(\pi/2) = 1$, $\cos(\pi/2) = 0.0000$ — the pattern sweeps the full unit circle as position advances.

**4. Why order becomes visible.** Each position gets a *different* fixed vector added, so two tokens' combined vectors now depend on their positions — the score $e_{ij} = (W_Q\mathbf{x}_{\text{pos }i})^\top(W_K\mathbf{x}_{\text{pos }j})$ changes when positions swap, breaking the permutation symmetry that made reordered sequences indistinguishable.

**5. Wavelength ladder.** Short wavelengths resolve *nearby* positions (adjacent tokens differ); long wavelengths avoid premature wrapping/aliasing so *distant* positions stay distinguishable over long ranges — a mixed-radix position code needs both scales to uniquely pin down any position in the context.

**6. Permutation argument.** Scores are $S = QK^\top$ over token vectors; permuting sequence rows permutes rows *and columns* of $S$ by the same permutation; row-wise softmax then permutes $A$ identically; $AV$'s rows permute — outputs change only by row order, i.e., no order-sensitive structure exists (set behavior).

## 02-attention.md

**1. Softmax by hand.** $\operatorname{softmax}(2, 1)$: $\frac{e^2}{e^2 + e^1} = \frac{7.389}{10.107} = 0.731$, other $= 0.269$. Scaled inputs $(1.414, 0.707)$: $\frac{4.113}{4.113 + 2.028} = 0.670$, other $= 0.330$ — **unscaled is sharper** ($0.731$ vs $0.670$): shrinking the score gap softens the distribution, which is exactly the point of $1/\sqrt{d_k}$ (bigger $d_k$ would otherwise sharpen automatically).

**2. Variance at $d_k = 9$.** Raw scores: $\operatorname{Var} = 9 \Rightarrow$ std $= \sqrt9 = 3$. Scaled by $\sqrt{d_k} = 3$: variance $9/9 = 1$, std $= 1$ — spread brought to unit scale regardless of width.

**3. Shapes (one head).** $X \in \mathbb{R}^{n \times 512}$; $Q = XW_Q \in \mathbb{R}^{n \times 64}$; $QK^\top \in \mathbb{R}^{n \times n}$; $A \in \mathbb{R}^{n \times n}$ (rows sum to 1); $AV \in \mathbb{R}^{n \times 64}$ ($d_v = 64$).

**4. Masked row 1.** Scores row 1: $(1, 0.5) + (0, -\infty) = (1, -\infty)$; $\exp(-\infty) = 0$ ⇒ $A_{1\cdot} = (1, 0)$ — token 1 attends only to itself (renormalization over the allowed entry).

**5. Why separate $W_Q$ and $W_K$.** Queries *seek* (what this token needs) while keys *advertise* (what each token offers); matching needs two different maps of the same $\mathbf{x}$ — sharing one matrix would force "questions" and "advertisements" into the same coordinate system, halving the space of learnable match functions (and removing the $W_Q^\top W_K$ asymmetry that scores directed relationships).

**6. $O = AV$ as an average.** Each output row is $\mathbf{o}_i = \sum_j A_{ij} \mathbf{v}_j$ — a convex combination (weights $A_{ij} \ge 0$, summing to 1 per row) of value vectors, mixing token $i$'s neighborhood according to its query's attention distribution.

## 03-transformer-block.md

**1. Multi-head shapes and parameters.** $d_k = 768/12 = 64$. Each of $W_Q, W_K, W_V, W_O$ is $768 \times 768 = 589{,}824$; combined $4 \times 589{,}824 = \mathbf{2{,}359{,}296}$ ($\approx 2.36$M) — matches $h$ heads of $64 \times 768$ each, per matrix.

**2. FFN count.** $768 \times 3072 + 3072 \times 768 = 2{,}359{,}296 + 2{,}359{,}296 = \mathbf{4{,}718{,}592}$ ($\approx 4.72$M) — note: nearly $2\times$ the whole attention parameter set of (1).

**3. LayerNorm of $(1, 3, 5)$.** $\mu = 3$; $\sigma^2 = \frac{(1-3)^2 + 0 + (5-3)^2}{3} = \frac{8}{3} \approx 2.667$; $\sigma \approx 1.633$; output $= (-1.225,\ 0,\ 1.225)$ — exactly the **z-scores** of section 04 (mean 0, variance 1), i.e., standardization applied to one token's feature vector.

**4. Gradient over 50 layers.** With residual: per-block factor $1 + 0.01 = 1.01$; $1.01^{50} \approx \mathbf{1.64}$ — signal survives and even grows mildly. Without: $0.01^{50} = 10^{-100} \approx \mathbf{0}$ — total vanishing (the failure residuals eliminate).

**5. Why alternate.** Attention mixes information *across positions* (who should talk to whom); the FFN transforms each position's content *independently — mixing without processing leaves raw retrieved signals unrefined, processing without mixing can't combine distant context.

**6. 32-block model arithmetic.** Per block: attention $4 \times 1024^2 = 4{,}194{,}304$; FFN $2 \times 1024 \times 4096 = 8{,}388{,}608$; total $12{,}582{,}912 \approx \mathbf{12.58}$ **M/block**; times 32: $\mathbf{402{,}653{,}184} \approx \mathbf{402.7}$ **M** parameters (before embeddings, output head, and norms).
