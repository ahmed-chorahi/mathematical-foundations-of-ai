# The Chain Rule

## Why It Matters for AI

A neural network is a composition: input → layer → layer → loss. To differentiate the loss with respect to an early weight, you must pass through every layer in between. The **chain rule** is the rule for differentiating compositions — and applying it layer by layer *is* backpropagation. If one chapter of this repository must be remembered, this is it.

## One Variable

If $y = f(u)$ and $u = g(x)$, then $y = f(g(x))$ and

$$
\boxed{\;\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}\;}
$$

"Rates multiply along the chain": the output moves $dy/du$ per unit of $u$, and $u$ moves $du/dx$ per unit of $x$ — so $y$ moves by the product per unit of $x$.

Equivalently: $(f \circ g)'(x) = f'(g(x))\cdot g'(x)$.

**Worked example 1.** $y = (3x + 1)^2$. Let $u = 3x+1$, $y = u^2$:

$$
\frac{dy}{du} = 2u, \quad \frac{du}{dx} = 3
\quad\Rightarrow\quad
\frac{dy}{dx} = 2u \cdot 3 = 6(3x+1)
$$

Check by expanding: $(3x+1)^2 = 9x^2 + 6x + 1$, derivative $18x + 6 = 6(3x+1)$ ✓

**Worked example 2.** $\dfrac{d}{dx}e^{2x^2}$: outer $e^u$, inner $u = 2x^2$:

$$
e^{2x^2}\cdot 4x
$$

**Worked example 3 — the pattern behind $\ln$ gradients.** $\dfrac{d}{dx}\ln(wx + b) = \dfrac{1}{wx+b}\cdot w = \dfrac{w}{wx+b}$.

The pattern to internalize: **differentiate the outer function, keep the inner argument, multiply by the derivative of the inner**.

## Several Variables

If $z = f(x, y)$ with $x = g(t)$ and $y = h(t)$, then along the path $t \mapsto (x(t), y(t))$:

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}
$$

Both paths contribute — contributions **add** (this is where the scalar chain rule's product generalizes: in one dimension there is a single path, so adding terms and multiplying coincide).

**Worked example.** $z = x^2 y$, $x = t$, $y = t^2$:

$$
\frac{\partial z}{\partial x} = 2xy = 2t^3, \quad
\frac{\partial z}{\partial y} = x^2 = t^2, \quad
\frac{dx}{dt} = 1, \quad \frac{dy}{dt} = 2t
$$

$$
\frac{dz}{dt} = 2t^3 \cdot 1 + t^2 \cdot 2t = 4t^3
$$

Check directly: $z = x^2 y = t^2 \cdot t^2 = t^4$, so $\dfrac{dz}{dt} = 4t^3$ ✓

**Multiple inner variables of the same input.** If $z = f(x,y)$ and *both* $x$ and $y$ depend on $t$, the two terms above both appear. If $z$ depends on a single intermediate $u(t)$, the rule collapses to the scalar case.

## Composition of Several Layers

For $L = f(g(h(x)))$ — a chain of three:

$$
\frac{dL}{dx} = \frac{dL}{dg}\cdot\frac{dg}{dh}\cdot\frac{dh}{dx}
$$

Multipliers accumulate along the path: **derivative of the outer × derivative of the next × … × derivative of the innermost**.

## Computational Graph

Write the computation as a DAG; each node applies one function. Forward pass computes values; backward pass multiplies local derivatives.

```text
x ──→ [u = 3x+1] ──→ [y = u²] ──→ L = y − 5
              (du/dx=3)      (dy/du=2u)   (dL/dy=1)
```

Backward: $\dfrac{dL}{dx} = \dfrac{dL}{dy}\cdot\dfrac{dy}{du}\cdot\dfrac{du}{dx} = 1 \cdot 2u \cdot 3 = 6u$.

At $x = 2$: $u = 7$, $dL/dx = 42$.

This bookkeeping — compute forward with intermediates cached, multiply local derivatives backward — is exactly how automatic differentiation works. Nothing more.

## The Full Form for Networks (Preview)

For a scalar loss $L$ depending on parameters through intermediate vectors, every path contributes and paths **sum**; within a path, derivatives **multiply**. Section 08 applies this to $z = W\mathbf{x} + \mathbf{b}$, $a = f(z)$, $L(\hat{y}, y)$ and derives the update rule for every weight. The scalar rule above is the complete logical content; the vector version only adds bookkeeping (Jacobians — chapter 04).

**Worked example — the shape of a regression gradient.** One prediction $\hat y = wx + b$, squared loss $L = (\hat y - y)^2$ with target $y$ fixed:

$$
\frac{\partial L}{\partial w}
= \frac{\partial L}{\partial \hat y}\cdot\frac{\partial \hat y}{\partial w}
= 2(\hat y - y)\cdot x
$$

$$
\frac{\partial L}{\partial b} = 2(\hat y - y)\cdot 1
$$

Read the structure: $2(\hat y - y)$ is the *error signal* from the loss; multiplying by $x$ (the input) routes it back to the weight that produced the prediction. Backpropagation is this routing, repeated through every layer. Section 08 makes it systematic.

## Connection to AI

- **Backpropagation = chain rule on a graph.** Section 08 derives it; frameworks implement it automatically from exactly the rules above.
- **Why depth complicates training.** Each layer contributes a multiplicative factor. Factors smaller than 1 repeatedly → **vanishing gradients** (signal dies before reaching early layers); factors larger than 1 repeatedly → **exploding gradients**. The multiplicative structure of the chain rule is the cause, which is why activation choices (ReLU's derivative of 1 on the positive side) and normalization were designed around it.
- **Gradients of every loss** — cross-entropy, MSE, KL — are chain-rule computations through the model's output stage (sections 07, 10).
- **Learning rates and gradient scales** connect to how many chain factors multiply: a deep product changes effective gradient magnitude.

## Common Mistakes

- **Adding instead of multiplying** along a chain: one path multiplies; multiple paths add.
- Dropping an inner derivative: $\frac{d}{dx}\sin(x^2) = \cos(x^2)\cdot 2x$, not $\cos(x^2)$.
- Forgetting that intermediate results must be evaluated at the right point: $f'(g(x))$, not $f'(\text{some other point})$.
- Differentiating with respect to a variable that is constant along the path (its derivative is 0 — the term vanishes, but only after being considered).
- Assuming independence: if two branches depend on the same upstream node, both branches contribute to that node's gradient (they add).

## Exercises

1. Differentiate $f(x) = (x^2 + 1)^3$.
2. Differentiate $g(x) = e^{-3x}$.
3. Let $z = xy$, $x = t^2$, $y = \sin t$. Find $\dfrac{dz}{dt}$ and check by substitution ($z = t^2 \sin t$ — use the product rule on that form).
4. For $L = (\hat y - y)^2$ with $\hat y = w_1 x$ (a tiny network), find $\dfrac{\partial L}{\partial w_1}$ as a product of two local derivatives.
5. Let $a = \sigma(w_1 x)$ and $L = (a - y)^2$ (one sigmoid unit, no output weights). Write $\dfrac{\partial L}{\partial w_1}$ as a product of three local derivatives along the path $L \leftarrow a \leftarrow w_1x \leftarrow w_1$ (structure only — leave $\sigma'$ symbolic).
6. Evaluate your expression from exercise 5 at $x = 1$, $w_1 = 0$, $y = 0$, using $\sigma(0) = 0.5$ and $\sigma'(z) = \sigma(z)(1-\sigma(z))$.
