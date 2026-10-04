# 06 — Information Theory

Information theory quantifies **uncertainty** and **communication**. Entropy measures how uncertain a distribution is; cross-entropy measures the cost of describing reality using the wrong distribution; KL divergence measures the gap between two distributions; mutual information measures how much one variable tells you about another. These four quantities are the vocabulary of AI training objectives: every classifier trained with cross-entropy is doing information theory, whether or not it says so.

## Contents

| File | Topic | AI application |
|---|---|---|
| [01-entropy.md](01-entropy.md) | surprise, entropy, bits and nats | measuring uncertainty |
| [02-cross-entropy.md](02-cross-entropy.md) | coding under a model, the loss | the classification loss |
| [03-kullback-leibler-divergence.md](03-kullback-leibler-divergence.md) | distance between distributions | distillation, likelihood ratios |
| [04-conditional-entropy.md](04-conditional-entropy.md) | uncertainty given knowledge | value of context |
| [05-mutual-information.md](05-mutual-information.md) | shared information between variables | feature relevance, information bottleneck |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Section 03 (probability, expectation — entropy *is* an expectation) and section 05 (minimizing losses — cross-entropy is a loss you already minimize there).

## Notation Conventions

- Logs: **$\log$ means $\ln$ (natural log) throughout** — measured in **nats**. When bits are wanted, $\log_2$ is written explicitly — measured in **bits**. The two differ only by the constant factor $\ln 2$, which is absorbed into learning rates during training; the *structure* of every result is identical.
- Distributions are discrete unless stated; $p$, $q$ are probability vectors over the same alphabet.
