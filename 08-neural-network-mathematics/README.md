# 08 — Neural Network Mathematics

A neural network is not a new kind of mathematics: it is linear algebra (weights as matrices), calculus (chain rule), probability (loss as likelihood), and optimization (gradient descent) stacked into one object. This section builds the network from a single neuron up, computes a full forward pass by hand, derives backpropagation from the chain rule with worked numbers, and closes with what training actually does — and where it can fail.

## Contents

| File | Topic | Tools reused |
|---|---|---|
| [01-neuron-and-activations.md](01-neuron-and-activations.md) | $z = w^\top x + b$, $a = f(z)$, activation functions, why nonlinearity | logistic regression, derivatives |
| [02-layers-and-forward-pass.md](02-layers-and-forward-pass.md) | $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, stacking layers, worked forward pass, loss | matrix multiplication, shapes |
| [03-backpropagation.md](03-backpropagation.md) | chain rule to gradients, worked backward pass, $\partial L/\partial W = \delta x^\top$ | chain rule (section 02) |
| [04-training-loop.md](04-training-loop.md) | SGD on networks, epochs, initialization, vanishing gradients, failure modes | SGD (section 05), BCE/MSE losses |

Exercises are at the end of each chapter; solutions are in [SOLUTIONS.md](SOLUTIONS.md).

## Prerequisites

Sections 01 (matrix multiplication), 02 (chain rule), 05 (gradient descent, SGD), 06 (cross-entropy), 07 (logistic regression — the single-neuron special case).

## The Shape of the Whole Story

$$
\mathbf{x} \xrightarrow{\ W_1,\,b_1\ } \mathbf{z}_1 \xrightarrow{\ f\ } \mathbf{a}_1 \xrightarrow{\ W_2,\,b_2\ } \cdots \xrightarrow{\ f\ } \hat{y}
\xrightarrow{\ L(\hat y, y)\ } \text{scalar}
\quad\xrightarrow{\ \text{backprop: chain rule}\ }\quad
\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}, \ldots
\xrightarrow{\ \text{SGD}\ }\ \text{repeat}
$$

Forward: compute the prediction (sections 01–02). Backward: propagate error signals to every weight (section 03). Update: descend (section 04 — already learned, now applied to millions of parameters).
