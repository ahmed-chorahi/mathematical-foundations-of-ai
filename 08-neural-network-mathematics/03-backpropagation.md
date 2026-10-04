# Backpropagation

## Why It Matters for AI

Training needs one thing mathematically: $\dfrac{\partial L}{\partial \theta}$ for *every* weight and bias in the network — potentially billions of them. **Backpropagation** computes all of them with one systematic sweep, by applying the chain rule backward through the computation graph. No new mathematics — section 02's chain rule, executed in the right order. This chapter derives the rules, then computes the worked example from chapter 02 completely, number by number.

## The Task

$L$ is a scalar; $\theta$ contains every parameter. We need $\nabla_\theta L$ once per training step. Forward direction: $\mathbf{x} \to \mathbf{z}_1 \to \mathbf{a}_1 \to \cdots \to \hat y \to L$. The chain rule connects derivatives along this path.

## Chain Rule Refresher (Section 02)

Single chain: $L(g(h(t)))$:

$$
\frac{dL}{dt} = \frac{dL}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dt}
$$

Multi-path: if $t$ reaches $L$ through several routes, **sum over paths** (the sum rule):

$$
\frac{dL}{dt} = \sum_{\text{paths}} \prod_{\text{edges on path}} \text{(edge derivative)}
$$

Two building blocks, applied repeatedly.

## The Two Local Rules

**Rule 1 — scalar chain through one operation.** For $u = g(t)$:

$$
\frac{\partial L}{\partial t} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial t}
$$

*Meaning:* you already know the upstream sensitivity $\partial L/\partial u$; multiply by the operation's local derivative to push it one step back.

**Rule 2 — vector/matrix layers.** For layer $\mathbf{z} = W\mathbf{a} + \mathbf{b}$ with upstream $\boldsymbol{\delta} = \dfrac{\partial L}{\partial \mathbf{z}} \in \mathbb{R}^m$:

$$
\boxed{\;
\frac{\partial L}{\partial W} = \boldsymbol{\delta}\, \mathbf{a}^\top \in \mathbb{R}^{m \times n},\qquad
\frac{\partial L}{\partial \mathbf{b}} = \boldsymbol{\delta},\qquad
\frac{\partial L}{\partial \mathbf{a}} = W^\top \boldsymbol{\delta}
\;}
$$

*Shapes:* $\boldsymbol{\delta}$ ($m$) outer-product $\mathbf{a}^\top$ ($n$) $\to$ $m \times n$ = shape of $W$ ✓.

*Why $\partial L/\partial \mathbf{a} = W^\top \boldsymbol{\delta}$ (the crucial one):* each row $a_j$ influences every output $z_i$ through $W_{ij}$, with local slope $W_{ij}$; sum over all paths (sum rule):

$$
\frac{\partial L}{\partial a_j} = \sum_{i=1}^m \frac{\partial L}{\partial z_i}\, W_{ij} = \big(W^\top \boldsymbol{\delta}\big)_j
$$

*Why $\partial L/\partial W = \boldsymbol{\delta}\mathbf{a}^\top$:* $z_i = \sum_j W_{ij}a_j + b_i$ gives $\partial z_i / \partial W_{ij} = a_j$, so $\partial L/\partial W_{ij} = \delta_i a_j$ — the $(i,j)$ entry of $\delta a_j$ ✓.

**The delta recurrence (the algorithm):** each layer's $\boldsymbol{\delta}$ = (local derivative of its nonlinearity, elementwise) × ($W^\top$ of the *next* layer's $\delta$):

$$
\boldsymbol{\delta}^{(\ell)} = \big(f'(\mathbf{z}^{(\ell)}) \odot W^{(\ell+1)\top} \boldsymbol{\delta}^{(\ell+1)}\big)
$$

Sweep $\ell$ from the output layer down to $1$ — that's **backpropagation**: reverse order, multiply local derivatives, apply Rule 2 at each layer.

## Full Worked Example (Chapter 02's Network)

Same network, input, target; $f = \sigma$ everywhere, BCE loss.

**Forward (from chapter 02).**

$$
\mathbf{a}^{(1)} = (0.7311,\ 0.8808),\qquad
z^{(2)} = -0.0749,\qquad
\hat y = 0.4813,\qquad
L = 0.6564
$$

**Step 1 — output layer delta.** For sigmoid + BCE, section 07 gave the collapse (the gift that keeps giving):

$$
\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} = \hat y - y = 0.4813 - 0 = \mathbf{0.4813}
$$

**Step 2 — output layer parameters** (Rule 2, $\mathbf{a} = \mathbf{a}^{(1)}$):

$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)}\,\mathbf{a}^{(1)\top}
= 0.4813 \cdot (0.7311,\ 0.8808) = (0.3519,\ 0.4239)
$$
$$
\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)} = 0.4813
$$

**Step 3 — push into the hidden layer:**

$$
\frac{\partial L}{\partial \mathbf{a}^{(1)}} = W^{(2)\top} \delta^{(2)}
= \begin{bmatrix}0.5\\-0.5\end{bmatrix} 0.4813 = \begin{bmatrix}0.2406\\-0.2406\end{bmatrix}
$$

**Step 4 — hidden delta** (multiply by $\sigma'$ elementwise; $\sigma'(z) = \sigma(z)(1-\sigma(z))$):

$$
\boldsymbol{\delta}^{(1)} = \mathbf{a}^{(1)} \odot (1 - \mathbf{a}^{(1)}) \odot \frac{\partial L}{\partial \mathbf{a}^{(1)}}
$$

- neuron 1: $0.7311 \times 0.2689 \times 0.2406 = \mathbf{0.0473}$
- neuron 2: $0.8808 \times 0.1192 \times (-0.2406) = \mathbf{-0.0253}$

**Step 5 — hidden layer parameters:**

$$
\frac{\partial L}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^\top
= \begin{bmatrix}0.0473\\-0.0253\end{bmatrix}\begin{bmatrix}1 & 2\end{bmatrix}
= \begin{bmatrix} 0.0473 & 0.0946 \\ -0.0253 & -0.0505 \end{bmatrix},
\qquad
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \begin{bmatrix}0.0473\\-0.0253\end{bmatrix}
$$

**Step 6 — one SGD update** ($\eta = 0.1$) on the output weights:

$$
W^{(2)} \leftarrow \begin{bmatrix}0.5 & -0.5\end{bmatrix} - 0.1\,(0.3519,\ 0.4239)
= \begin{bmatrix}0.4648 & -0.5424\end{bmatrix},\qquad
b^{(2)} \leftarrow -0.0481
$$

All nine parameters receive gradients; all nine move (full arrays above; every number verified by execution).

```python
# backward pass of the worked example
d2 = 0.4813                      # y_hat - y
dL_dw2 = [d2 * 0.7311, d2 * 0.8808]           # (0.3519, 0.4239)
dL_da1 = [d2 * 0.5, d2 * -0.5]                # (0.2406, -0.2406)
a1 = [0.7311, 0.8808]
d1 = [a1[i]*(1-a1[i])*dL_da1[i] for i in range(2)]   # (0.0473, -0.0253)
# dL/dW1 = outer(d1, x) with x = (1, 2)
```

## What Backprop Is (and Is Not)

- **Is:** dynamic programming over the computation graph — each edge's derivative computed once, reused by every path through it (the sum rule handled incrementally). One backward sweep = gradients for *all* parameters, at cost ≈ 2× the forward pass.
- **Is not:** an approximation, an alternative to the chain rule, or a guarantee of good results — it gives *exact* gradients of the scalar $L$ w.r.t. $\theta$. Whether SGD then finds a *useful* $\theta$ is chapter 04's honest discussion (non-convexity — section 05 again).
- **Cost:** time ≈ constant-factor of forward; memory = storing activations (the forward pass's $\mathbf{z}, \mathbf{a}$) until the backward sweep finishes — frameworks' `no_grad`/checkpointing exist for this trade.
- **Automatic differentiation:** every framework implements exactly this procedure mechanically over a graph of ops — hand-derived here once, automated everywhere.

## Connection to AI

- **Transformers train with this** — attention, softmax, layer norm all have local derivatives; the sweep is identical (section 09's formulas are forward-pass shapes; their gradients flow through the same Rule 1/2 machinery).
- **Vanishing/exploding gradients are this chapter's multiplicative cascade:** $\boldsymbol{\delta}^{(\ell)}$ multiplies $f'$ and $W^\top$ at each step down; small $f'$ (sigmoid, chapter 01) ⇒ decay; large singular values of $W$ ⇒ growth. Initialization, normalization, and residual paths all aim at keeping this product well-scaled.
- **Gradient checks:** the standard way to validate an implementation — compare backprop's $\partial L/\partial\theta$ with finite differences $\frac{L(\theta + h) - L(\theta - h)}{2h}$ (section 02's derivative *definition*); mismatch ⇒ bug in shapes or a missing term.
- **Second-order methods** (Hessian-based optimizers) would need backprop of backprop — possible (hypergradients), rarely worth it; first-order SGD dominates.

## Common Mistakes

- Updating weights **before** the backward pass — gradients would be computed for stale parameters; order: forward → backward → update → zero grads.
- Forgetting to zero gradients between steps — accumulation across batches (sometimes deliberate for large effective batches — careful, it's a feature *and* a footgun).
- Wrong shapes in $\boldsymbol{\delta}\mathbf{a}^\top$ (transposed convention) — the #1 implementation bug; check against the shape table.
- Missing the elementwise $f'$ factor in the hidden delta (linear layers would use just $W^\top \delta$).
- Using $\hat y - y$ where the loss *isn't* sigmoid+BCE (MSE output's delta is $2(\hat y - y)/n$-scaled, etc. — the upstream factor depends on the loss).
- Computing gradients on a tensor that doesn't require grad (detached) — silently zero gradients; a framework symptom of a graph break.

## Exercises

1. Scalar chain: $L = a^2$, $a = \sigma(z)$, $z = wx + b$. Derive $\dfrac{\partial L}{\partial w}$ and $\dfrac{\partial L}{\partial b}$ symbolically; evaluate at $x = 2$, $w = 0$, $b = 0$.
2. Rule 2 shapes: verify $\boldsymbol{\delta}\mathbf{a}^\top$ has $W$'s shape for $\boldsymbol{\delta} \in \mathbb{R}^3$, $\mathbf{a} \in \mathbb{R}^5$ — what is $W$'s shape, and $\partial L/\partial \mathbf{a}$'s?
3. Reuse the worked example, but with target $y = 1$ (same forward numbers): recompute $\delta^{(2)}$ and $\partial L / \partial W^{(2)}$.
4. Hidden-neuron-1 gradient check: confirm $0.7311 \times 0.2689 \times 0.2406 \approx 0.0473$ (compute).
5. Take the output-layer update from Step 6 with $\eta = 0.1$, re-run the forward pass, and check whether $L$ decreased (compute the new $\hat y$ and $L$).
6. State in one sentence why backprop costs about one forward pass in time but requires stored activations in memory.
