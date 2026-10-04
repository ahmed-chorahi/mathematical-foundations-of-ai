# 08 — Neural Network Mathematics: Solutions

## 01-neuron-and-activations.md

**1. Collapse of two affine maps.**
$$
\mathbf{z}_2 = W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\,\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2)
$$
One effective weight matrix $W' = W_2 W_1$ and one effective bias $\mathbf{b}' = W_2 \mathbf{b}_1 + \mathbf{b}_2$ — an affine map of $\mathbf{x}$, however many layers deep.

**2. Values at $z = 0$.** Sigmoid: $f = 0.5$, $f' = 0.5 \times 0.5 = 0.25$. Tanh: $f = 0$, $f' = 1 - 0 = 1$. ReLU: $f = 0$; derivative undefined at $0$ (subdifferential $[0, 1]$ — frameworks pick a convention, typically $0$ or $1$).

**3. Twenty sigmoid layers.** Worst case: $(0.25)^{20} \approx 9.1 \times 10^{-13}$ — gradients of the earliest layers are numerically zero; nothing learns below a certain depth (vanishing gradients; hence ReLU-family activations).

**4. Price prediction head.** **Identity (linear) output + MSE** (or Gaussian NLL): prices are unbounded reals — sigmoid would cap outputs in $(0,1)$, tanh in $(-1,1)$; both would require rescaling and cannot extrapolate. (Log-link variants exist in practice; plain identity is the textbook answer.)

**5. $\tanh(1)$ and $\tanh'(1)$.** $\tanh(1) \approx 0.7616$; $\tanh'(1) = 1 - 0.7616^2 \approx 1 - 0.5800 = 0.4200$.

**6. Dying ReLU and mitigation.** If a neuron's pre-activations stay negative (poor init, large learning rate shifting biases), it outputs $0$ with gradient $0$ forever — permanently dead. Mitigations: leaky ReLU ($\max(0.01z, z)$) keeps a nonzero slope for $z < 0$; also sensible initialization and monitoring activation statistics.

## 02-layers-and-forward-pass.md

**1. Shapes and parameter count.**

| tensor | shape | parameters |
|---|---|---|
| $W^{(1)}$ | $128 \times 784$ | $100{,}352$ |
| $\mathbf{b}^{(1)}$ | $128$ | $128$ |
| $W^{(2)}$ | $64 \times 128$ | $8{,}192$ |
| $\mathbf{b}^{(2)}$ | $64$ | $64$ |
| $W^{(3)}$ | $10 \times 64$ | $640$ |
| $\mathbf{b}^{(3)}$ | $10$ | $10$ |

Total: $100{,}352 + 128 + 8{,}192 + 64 + 640 + 10 = \mathbf{109{,}386}$.

**2. Order matters.** $W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = (3, 2) + (0.5, -0.5) = (3.5, 1.5)$. $W^{(1)}(\mathbf{x} + \mathbf{b}^{(1)})$: $\mathbf{x} + \mathbf{b}^{(1)} = (1.5, 1.5)$, then $W^{(1)}(1.5, 1.5) = (3, 1.5)$. **Different** ($(3.5, 1.5) \ne (3, 1.5)$) — bias shifts the *input*, not the *output*; the composition of affine maps requires the defined order $W(\cdot) + \mathbf{b}$.

**3. Forward pass variant.** $\mathbf{a}^{(1)} = (\sigma(1), \sigma(1)) = (0.7311, 0.7311)$; $z^{(2)} = 0.7311 + 0.7311 - 2 = -0.5379$; $\hat y = \sigma(-0.5379) \approx \mathbf{0.3687}$.

**4. BCE with $y = 1$.** $L = -\ln \hat y = -\ln(0.3687) \approx 0.9978$ nats — nearly $1$ nat for a wrong-leaning prediction.

**5. Collapse check.** Stepwise: $W^{(1)}\mathbf{x} = (1, 0)$; $+ \mathbf{b}^{(1)} = (2, 0)$; $W^{(2)}(2, 0) + \mathbf{b}^{(2)} = (4, 2) + (0, 1) = \mathbf{(4, 3)}$.
Combined: $W' = W^{(2)}W^{(1)} = \begin{bmatrix}2&4\\1&3\end{bmatrix}$, $\mathbf{b}' = W^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)} = (2, 1) + (0, 1) = (2, 2)$; $W'\mathbf{x} + \mathbf{b}' = (2, 1) + (2, 2) = \mathbf{(4, 3)}$ ✓.

**6. Why activations are stored.** Every local derivative in the backward pass is evaluated at the *forward* values ($\sigma'(\mathbf{z})$ needs $\mathbf{z}$; $\partial L/\partial W = \delta \mathbf{a}^\top$ needs $\mathbf{a}$) — those intermediates must be kept until the backward sweep consumes them (memory is the price of gradients).

## 03-backpropagation.md

**1. Scalar chain.** $\dfrac{\partial L}{\partial w} = 2a \cdot \sigma'(z) \cdot x$, $\dfrac{\partial L}{\partial b} = 2a \cdot \sigma'(z)$. At $x = 2$, $w = b = 0$: $z = 0$, $a = 0.5$, $\sigma'(0) = 0.25$:

$$
\frac{\partial L}{\partial w} = 2(0.5)(0.25)(2) = 0.5, \qquad
\frac{\partial L}{\partial b} = 2(0.5)(0.25) = 0.25
$$

**2. Shapes.** $\boldsymbol{\delta}\mathbf{a}^\top \in \mathbb{R}^{3} \times \mathbb{R}^{5} = \mathbb{R}^{3 \times 5}$ ⇒ $W \in \mathbb{R}^{3 \times 5}$; $\partial L / \partial \mathbf{a} = W^\top \boldsymbol{\delta} \in \mathbb{R}^{5}$.

**3. Target $y = 1$.** $\delta^{(2)} = 0.4813 - 1 = -0.5187$; $\partial L / \partial W^{(2)} = -0.5187 \times (0.7311, 0.8808) = \mathbf{(-0.3792, -0.4569)}$ — signs flipped vs. $y = 0$: now the output weights push $\hat y$ *up* toward 1.

**4. Hidden-neuron-1 gradient.** $0.7311 \times 0.2689 \times 0.2406 \approx 0.1966 \times 0.2406 \approx \mathbf{0.0473}$ ✓ (matches Step 4).

**5. Loss after one update.** New $z^{(2)} = 0.4648(0.7311) - 0.5424(0.8808) - 0.0481 = -0.1861$; $\hat y = \sigma(-0.1861) = 0.4536$; $L = -\ln(1 - 0.4536) = 0.6044 < 0.6564$ — **the loss decreased** ✓ (the step moved against the gradient, as the theory promised).

**6. Time vs memory.** Backward reuses each operation's cached forward inputs/outputs to evaluate local derivatives, so those intermediates must exist in memory until the sweep completes — one forward pass in time (≈2× with backward), stored activations in memory.

## 04-training-loop.md

**1. Pseudocode.**
```text
for epoch = 1..3:
    shuffle data
    for each minibatch B:
        forward:  L = loss(net(x_B), y_B)
        backward: g = ∇L(B)          # backprop, chapter 03
        update:   θ ← θ − η·g        # SGD, section 05
        zero:     g ← 0
```
(Forward → backward → update → zero, in that order, every iteration.)

**2. Overfitting interventions.** (a) **Early stopping** — checkpoint at the validation minimum; (b) **weight decay / dropout** — section 05's penalties; (c) **data augmentation or more data** — shrink the estimation gap; alternatives: smaller capacity, looser feature set. Any three.

**3. Zero-init collapse.** All neurons start with identical weights ⇒ identical outputs on every input ⇒ identical gradients ⇒ updated identically ⇒ stay identical forever — width contributes no diversity; symmetry must be broken by random initialization.

**4. Surviving factors.** Sigmoid: $(0.25)^8 = 1/65536 \approx \mathbf{1.5 \times 10^{-5}}$ — negligible. ReLU on active paths: $1^8 = \mathbf{1}$ — the full gradient survives (the activation's whole rationale).

**5. Randomness and reproducibility.** (i) Weight **initialization**; (ii) **minibatch order** (shuffling each epoch); (iii) **dropout** masks (when used). Fix: fix all random seeds (init, data order, dropout) — plus deterministic kernels where available.

**6. What backprop guarantees.** *Guarantees:* exact partial derivatives of the computed loss $L$ w.r.t. every parameter (the chain rule, evaluated correctly). *Does not guarantee:* that SGD reaches a global minimum, or that the resulting model generalizes — non-convexity and estimation remain open questions (chapters on honesty above).
