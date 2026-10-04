# Functions and Graphs

## Why It Matters for AI

A model is a function: inputs in, outputs out. Training changes the function's parameters. Loss functions, activation functions, and the softmax that produces token probabilities are all functions, and the language of calculus — limits, derivatives, continuity — applies to functions first. This chapter fixes the vocabulary.

## Definition

A **function** $f$ from a set $A$ to a set $B$ assigns to each element $x$ of $A$ exactly one element $f(x)$ of $B$.

- $A$ is the **domain** — the set of allowed inputs.
- $B$ is the **codomain** — the set of possible outputs.
- The **range** is the set of values actually attained: $\{f(x) : x \in A\}$.
- $x$ is the (independent) variable, $f(x)$ is the value, $f$ is the rule.

"$f: \mathbb{R} \to \mathbb{R}$" reads "f maps real numbers to real numbers". Written with an unnamed input, $y = f(x)$; applied to a specific value, $f(2) = 4$ if $f(x) = x^2$.

A **scalar-valued function** returns one number. A **vector-valued function** returns a vector: $\mathbf{f}(x) = (x, x^2)$ traces a parabola in the plane. Neural-network layers are vector-valued functions — a vector in, a vector out.

## Evaluating and Composing

**Composition** $ (g \circ f)(x) = g(f(x)) $ applies $f$ first, then $g$. Layers of a neural network are a composition: $ \mathbf{a}^{(3)} = f_3\big(f_2(f_1(\mathbf{x}))\big) $. The chain rule of calculus (section 02) exists to differentiate exactly this structure.

**Worked example.** Let $f(x) = 2x + 1$ and $g(x) = x^2$. Then

$$
g(f(3)) = g(7) = 49, \qquad f(g(3)) = f(9) = 19.
$$

Composition is not commutative: $g(f(3)) \ne f(g(3))$.

## Inverse Functions

$f^{-1}$ undoes $f$: if $y = f(x)$ then $x = f^{-1}(y)$. An inverse exists exactly when $f$ is **one-to-one** (injective) — no two inputs share an output.

Logarithms are the inverse of exponentiation because $2^x$ is one-to-one: $2^x = 8 \Rightarrow x = \log_2 8 = 3$.

Invertibility decides whether a linear system can be solved for its inputs (section 01: the matrix inverse) and whether an encoding can be decoded.

## Key Property: Continuity

Informally, $f$ is **continuous** at $a$ if $f(x)$ approaches $f(a)$ as $x$ approaches $a$ — no jumps, no gaps. Formally:

$$
\lim_{x \to a} f(x) = f(a)
$$

The **limit** $\lim_{x \to a} f(x) = L$ means: as $x$ gets arbitrarily close to $a$ (without equaling $a$), $f(x)$ gets arbitrarily close to $L$.

Example: $\lim_{x \to 0} \frac{\sin x}{x} = 1$ (a famous limit, used when deriving derivatives of trigonometric functions). Counter-example: the function that is $0$ at $x = 0$ and $1$ elsewhere has no limit at $0$ — it jumps.

Why it matters now: derivatives (section 02) are defined only where a function is continuous, and every activation function you meet will be continuous and usually smooth.

## Common Functions

| function | formula | shape | where it appears in AI |
|---|---|---|---|
| identity | $f(x) = x$ | straight line, slope 1 | residual connections |
| linear | $f(x) = ax + b$ | straight line, slope $a$ | a neuron without activation |
| square | $f(x) = x^2$ | parabola | squared-error loss, L2 regularization |
| reciprocal | $f(x) = 1/x$ | hyperbola | learning-rate intuition |
| exponential | $f(x) = e^x$ | growth, always positive | softmax, entropy, weight decay |
| logarithm | $f(x) = \log x$ | inverse of $e^x$, undefined at 0 | log-likelihood, cross-entropy |
| sigmoid | $f(x) = \dfrac{1}{1+e^{-x}}$ | S-shape in $(0,1)$ | logistic regression, gates |
| hyperbolic tangent | $\tanh x$ | S-shape in $(-1,1)$ | activations |
| ReLU | $f(x) = \max(0, x)$ | flat then rising | default neural-network activation |

The last three are **activation functions** — the nonlinear part of a neural network. Without them, stacking layers would collapse to a single linear map (an exercise in section 08 shows this).

## Graphs

The **graph** of $y = f(x)$ is the set of points $(x, f(x))$ in the plane. Reading a graph gives instant qualitative information:

- **slope** of a straight line = rate of change (and the derivative of a linear function),
- **steepness** = how fast output responds to input,
- **height** = function value,
- **intercepts** = special values: $f(0)$ (vertical intercept), $f(x) = 0$ (roots).

**Worked example.** Sketch $f(x) = 2x - 1$ mentally: a line through $(0, -1)$ with slope 2, so $x$-intercept solves $2x - 1 = 0 \Rightarrow x = 0.5$. The graph crosses the x-axis at $(0.5, 0)$.

For nonlinear functions, three anchor points are usually enough to place the shape: one root, the intercept, and one point far out.

## Piecewise Functions

A function defined by different rules on different parts of the domain:

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \le 0 \end{cases}
$$

Piecewise definitions are common in AI (ReLU, clipping, top-$k$ selection). Check continuity at the boundaries: ReLU's two pieces both give $0$ at $x = 0$, so it is continuous — but it has a corner there, and the derivative does not exist exactly at $x = 0$. Practically this is irrelevant; the gradient is defined almost everywhere.

## Connection to AI

- **A trained model is a function** $f_\theta: \mathbf{x} \mapsto \mathbf{y}$ with parameters $\theta$ inside the rule. Training = choosing $\theta$.
- **A loss is a function of the parameters**: $J(\theta)$ maps parameters to a single error number. Minimizing it is the whole training problem (section 05).
- **Composition = depth.** Deep learning is function composition, and backpropagation is the chain rule applied along the composition.
- **Piecewise = ReLU networks.** A ReLU network is piecewise linear: different regions of input space follow different linear functions.

## Common Mistakes

- Treating $f(x)$ (a value) and $f$ (the rule) as the same thing. $f$ is the map; $f(x)$ is where $x$ lands.
- Forgetting domain restrictions: $\log x$ requires $x > 0$, $1/x$ requires $x \neq 0$.
- Assuming $f(g(x)) = g(f(x))$. Composition order is fixed by causality: the input must go through the first layer first.
- Reading a graph's steepness as a value. Steepness is a rate (a future derivative); height is the value.

## Exercises

1. Let $f(x) = 3x - 2$ and $g(x) = x^2 + 1$. Compute $f(2)$, $g(f(1))$, and $f(g(1))$.
2. State the domain of $f(x) = \dfrac{1}{\sqrt{x}}$.
3. Show that $f(x) = 2x + 5$ is one-to-one and find $f^{-1}(y)$.
4. Evaluate $\lim_{x \to 2} (x^2 - 1)$ and explain why no limit laws were needed.
5. Compute ReLU$(-3)$ and ReLU$(-3 + 10^{-6})$ using the piecewise definition.
6. The sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ satisfies $\sigma(-x) = 1 - \sigma(x)$. Verify this algebraically for any $x$.
