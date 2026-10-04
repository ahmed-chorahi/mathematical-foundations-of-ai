# Layers and the Forward Pass

## Why It Matters for AI

One neuron computes a number; a **layer** computes a vector at once — a single matrix product. Stacking layers and applying activations is the **forward pass**: the function the network actually is, from input to prediction to loss. This chapter fixes notation, shapes, and one complete worked example you can follow on paper; every later chapter (backprop, transformers) uses exactly this machinery.

## A Layer = A Matrix Product

$m$ neurons on the same inputs: collect their weights as **rows** of $W \in \mathbb{R}^{m \times d}$, biases as $\mathbf{b} \in \mathbb{R}^m$:

$$
\boxed{\;\mathbf{z} = W \mathbf{x} + \mathbf{b} \in \mathbb{R}^m,\qquad
\mathbf{a} = f(\mathbf{z}) \in \mathbb{R}^m\;}
$$

($f$ applied elementwise: $a_j = f(z_j)$.) Row $j$ of $W$ is neuron $j$'s weight vector — the same $\mathbf{w}^\top \mathbf{x} + b$ as before, done $m$ times in one operation (section 01's matrix–vector product, now earning its keep).

**Shape discipline** (the practical skill): $W \in \mathbb{R}^{m\times d}$ times $\mathbf{x} \in \mathbb{R}^{d}$ gives $\mathbb{R}^{m}$ — inner dimensions must match; every bug in real code starts with a shape mismatch.

## Stacking: The Network as Composition

A multilayer perceptron (MLP) with hidden sizes $d \to h_1 \to h_2 \to \cdots \to 1$:

$$
\mathbf{a}^{(0)} = \mathbf{x},\qquad
\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)},\qquad
\mathbf{a}^{(\ell)} = f\!\big(\mathbf{z}^{(\ell)}\big)
$$

one line per layer, chained: a **composition of affine maps and elementwise nonlinearities** — the class of functions called *deep neural networks* (shallow: one hidden layer; deep: several).

**Parameter count** = every weight and bias: layer $\ell$ with $n_{\ell-1} \to n_\ell$ units has $n_\ell \cdot n_{\ell-1} + n_\ell$ parameters. (Convolutional/recurrent weight-sharing reduces the *effective* count — beyond this section.)

## Worked Forward Pass (2–2–1, All Sigmoid)

Network: $d = 2$ inputs, one hidden layer of $2$ sigmoid neurons, one sigmoid output.

$$
W^{(1)} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\
\mathbf{b}^{(1)} = \begin{bmatrix}0\\0\end{bmatrix},\qquad
W^{(2)} = \begin{bmatrix} 0.5 & -0.5 \end{bmatrix},\
b^{(2)} = 0
$$

Input $\mathbf{x} = (1, 2)^\top$, target $y = 0$.

**Layer 1.**

$$
\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} = \begin{bmatrix}1&0\\0&1\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}1\\2\end{bmatrix},
\qquad
\mathbf{a}^{(1)} = \begin{bmatrix}\sigma(1)\\ \sigma(2)\end{bmatrix} = \begin{bmatrix}0.7311\\ 0.8808\end{bmatrix}
$$

**Output layer.**

$$
z^{(2)} = \begin{bmatrix}0.5 & -0.5\end{bmatrix}\begin{bmatrix}0.7311\\0.8808\end{bmatrix} + 0 = 0.5(0.7311) - 0.5(0.8808) = -0.0749
$$
$$
\hat y = a^{(2)} = \sigma(-0.0749) = 0.4813
$$

**Loss** (BCE with $y = 0$, section 06):

$$
L = -\big[y \ln \hat y + (1-y)\ln(1 - \hat y)\big] = -\ln(1 - 0.4813) = -\ln(0.5187) = 0.6564 \text{ nats}
$$

The network "thinks" the probability of class 1 is $0.4813$ — barely better than a coin, on a target of 0. Training will fix that (chapter 03 computes exactly how each weight should move).

**Parameters:** $W^{(1)}$ (4) + $\mathbf{b}^{(1)}$ (2) + $W^{(2)}$ (2) + $b^{(2)}$ (1) = **9 parameters** — all of them get gradients, all of them move in training.

```python
sig = lambda z: 1/(1 + __import__('math').exp(-z))
x = [1, 2]
a1 = [sig(x[0]), sig(x[1])]              # W1 = I, b1 = 0
z2 = 0.5*a1[0] - 0.5*a1[1]
y_hat = sig(z2)                          # 0.4813
from math import log
L = -log(1 - y_hat)                      # 0.6564 nats (y = 0)
```

## The Forward Pass in General

**Definition:** evaluating the network's function — every layer's $\mathbf{z}, \mathbf{a}$ in order — ending in the scalar loss $L(\hat y, y)$.

- **As a function:** $L_\theta(\mathbf{x}, y)$ where $\theta$ is the collection of *all* weights and biases. Training (chapters 03–04) needs $\partial L / \partial\theta$ for every component of $\theta$.
- **As a computation graph:** each node is an operation (multiply, add, activate) with cached inputs; backprop (chapter 03) walks this graph backward. The forward pass *stores* what the backward pass will need — memory is the price (activations of every layer kept until gradients are computed).
- **Batch form:** $\mathbf{X} \in \mathbb{R}^{n \times d}$ (rows = examples) gives $Z = \mathbf{X} W^\top + \mathbf{1}\mathbf{b}^\top$ — same math, parallelized; frameworks implement layers this way (section 05's minibatching lives here).

## Connection to AI

- **Width vs depth:** more neurons per layer = more units in the *approximation* (universal approximation: one hidden layer with enough neurons can approximate continuous functions on compact sets — an existence result with no efficiency guarantee); more layers = compositional structure — reusing sub-computations, which is why deep nets generalize with fewer total parameters on structured data (honest empirical fact, no complete theory).
- **Every architecture is this pattern with variants:** convolution = a layer with shared sparse $W$; attention = a layer whose $W$ depends on the input (section 09); an LLM = many such layers with residual connections.
- **Shapes predict parameters:** naming $d, h, V$ (vocab) etc. lets you size models on paper — the skill behind "GPT-style model with X billion parameters" arithmetic (section 10).
- **Loss placement is universal:** whatever the architecture, the last operation is a scalar loss comparable to labels — chapters 03–04 need nothing else.

## Common Mistakes

- Transposing $W$'s convention once and not elsewhere (rows-as-neurons vs columns-as-neurons) — pick one (rows here), state it, stay consistent.
- Forgetting biases (or double-counting: biases are *not* regularized by weight decay, section 05).
- Applying activation *before* the affine map ($f(W\mathbf{x})$ instead of $f(W\mathbf{x} + \mathbf{b})$) — wrong order, different function class.
- Losing track of elementwise-vs-matrix operations: $f$ acts elementwise; $W$ acts as a linear map.
- Assuming the worked example's identity $W^{(1)}$ is typical — it was chosen for arithmetic; real weights are small random numbers (chapter 04).
- Computing loss without the target's likelihood pairing (BCE for binary, CE for multiclass, MSE for regression — section 07's table).

## Exercises

1. Write the shapes for a network $784 \to 128 \to 64 \to 10$ (no activation detail): sizes of all $W^{(\ell)}, \mathbf{b}^{(\ell)}$ and the total parameter count.
2. Order of operations matters: with $W^{(1)} = \begin{bmatrix}1&1\\0&1\end{bmatrix}$, $\mathbf{b}^{(1)} = (0.5, -0.5)^\top$, $\mathbf{x} = (1,2)$ — compute $W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$ and $W^{(1)}(\mathbf{x} + \mathbf{b}^{(1)})$; do they agree?
3. Forward pass variant: $W^{(1)} = \begin{bmatrix}1&0\\0&1\end{bmatrix}$, $\mathbf{b}^{(1)} = \mathbf{0}$, $W^{(2)} = \begin{bmatrix}1 & 1\end{bmatrix}$, $b^{(2)} = -2$, $\mathbf{x} = (1,1)$. Compute $\hat y$ (sigmoid output).
4. With that $\hat y$ and $y = 1$, compute the BCE loss.
5. Verify the collapse claim numerically for $W_1 = \begin{bmatrix}1&2\\0&1\end{bmatrix}$, $\mathbf{b}_1 = (1,0)$, $W_2 = \begin{bmatrix}2&0\\1&1\end{bmatrix}$, $\mathbf{b}_2 = (0,1)$, $\mathbf{x} = (1,0)$: compute stepwise and via the combined $W' = W_2 W_1$, $\mathbf{b}' = W_2\mathbf{b}_1 + \mathbf{b}_2$.
6. Why does the forward pass store intermediate activations? (One sentence, keyword: backward.)
