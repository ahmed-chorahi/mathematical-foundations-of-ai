# 10 — LLM Mathematics

The endpoint. A large language model is a transformer (section 09) plus a probability objective (sections 03–07): it turns a token prefix into a distribution over the next token, is trained by cross-entropy on that conditional, is judged by perplexity (exponentiated entropy), and generates by sampling its own outputs at a chosen temperature. Every formula in this section was built earlier; this is their assembly into the object you actually use.

## Contents

| File | Topic | Tools reused |
|---|---|---|
| [01-pipeline-and-next-token.md](01-pipeline-and-next-token.md) | tokens → logits → softmax → $P(w_{t+1} \mid w_{\le t})$ | sections 07, 09 |
| [02-training-objective.md](02-training-objective.md) | sequence cross-entropy, factorization, teacher forcing | sections 04, 06, 07 |
| [03-perplexity.md](03-perplexity.md) | perplexity $= \exp(\text{loss})$, interpretation, limits | section 06's entropy |
| [04-sampling-and-temperature.md](04-sampling-and-temperature.md) | temperature, top-$p$, generation mechanics | section 06's entropy, softmax |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Sections 03 (conditionals), 06 (cross-entropy, entropy), 07 (softmax-classifier pipeline), 09 (transformer forward pass).

## Notation

- Sequence $w_1, w_2, \ldots, w_T$ (token IDs); $w_{\le t} = (w_1, \ldots, w_t)$ — the *prefix*.
- $V$ = vocabulary size; $\ell_t \in \mathbb{R}^V$ = logits at position $t$; $P_t = \operatorname{softmax}(\ell_t) \in \mathbb{R}^V$.
- Default logs are natural (nats) unless "bits" is written — section 06's convention, unchanged.
