# 09 — Transformer Mathematics

The transformer is a composition of the exact tools this course has built: embeddings are matrix rows (section 01), attention is scaled dot products plus softmax (sections 00, 03), causal masking is conditional structure (section 04's chain rule), layer normalization is the z-score (section 04), residuals keep gradients alive (section 08), and the whole block stack is trained by backprop with cross-entropy (sections 06–08). This section derives the forward pass formula by formula.

## Contents

| File | Topic | Tools reused |
|---|---|---|
| [01-embeddings-and-position.md](01-embeddings-and-position.md) | token vectors, sinusoidal position encodings, shapes | one-hot, matrix lookup |
| [02-attention.md](02-attention.md) | $Q,K,V$, scaled dot-product attention, $1/\sqrt{d_k}$ derived, causal mask | dot products, softmax, variance |
| [03-transformer-block.md](03-transformer-block.md) | multi-head attention, residuals, layer norm, feed-forward block | Section 08, z-scores |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Sections 01 (matrices, dot products), 04 (standardization), 05 (residual-style gradient arguments live in section 08), 06 (softmax as a distribution), 08 (layers, gradients).

## Notation

- $n$ = sequence length (tokens in context), $d_{\text{model}}$ = token vector width, $h$ = number of heads, $d_k$ = per-head key/query width ($d_k = d_{\text{model}}/h$ when heads partition the width), $V$ = vocabulary size.
- Matrices act on column vectors; a batch of sequences is a matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$ with **one token per row** (row convention of section 08).
