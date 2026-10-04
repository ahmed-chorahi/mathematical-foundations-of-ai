# Partial Derivatives and the Gradient

## Why It Matters for AI

A model has thousands of parameters at once. To know how the loss responds, you differentiate with respect to each parameter while holding the others fixed — that is a partial derivative — and collect the results into one vector: the **gradient**. The gradient is the central object of training: its direction is the direction of steepest increase of the loss, and its negative is the direction of descent.

## Functions of Several Variables

$f(x_1, \ldots, x_n)$ assigns one number to a vector of inputs:

$$
f: \mathbb{R}^n \to \mathbb{R}
$$

**Examples.** $f(x, y) = x^2 + 3xy$; the loss of a model $J(\theta_1, \ldots, \theta_d)$ as a function of all parameters; the height of a surface above the $(x,y)$ plane.

The output is one number — a **scalar field**. (Functions returning vectors are handled by the Jacobian in chapter 04.)

## Partial Derivative

The **partial derivative** with respect to $x$ treats all other variables as constants:

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}
$$

Written $f_x$ for short. Procedure: differentiate with respect to the chosen variable; every other variable is a constant coefficient.

**Worked example.** $f(x, y) = x^2 + 3xy + y^3$:

$$
\frac{\partial f}{\partial x} = 2x + 3y
\qquad
\frac{\partial f}{\partial y} = 3x + 3y^2
$$

Check the first: with $y$ fixed, $x^2 \to 2x$, $3xy \to 3y$ (times 1), $y^3 \to 0$ ✓

**Evaluated at a point:** $\dfrac{\partial f}{\partial x}\Big|_{(1,2)} = 2(1) + 3(2) = 8$ — near $(1,2)$, each unit of $x$ (y fixed) adds about $8$ to the output.

Each partial is an ordinary single-variable derivative in disguise — a slice through the multivariable surface.

## The Gradient

Stack the partials into a column vector:

$$
\nabla f = \begin{bmatrix} \partial f/\partial x_1 \\ \vdots \\ \partial f/\partial x_n \end{bmatrix}
\in \mathbb{R}^n
$$

$\nabla$ ("nabla") is the gradient operator. For $f(x,y) = x^2 + 3xy$:

$$
\nabla f = \begin{bmatrix} 2x + 3y \\ 3x \end{bmatrix},
\qquad
\nabla f\big|_{(1,2)} = \begin{bmatrix} 8 \\ 3 \end{bmatrix}
$$

The gradient is a **vector-valued function of the parameters**: at every point it produces one direction.

## What the Gradient Means

Two facts give the gradient its role in AI.

**1. Direction of steepest ascent.** For small step $\mathbf{u}$ with $\|\mathbf{u}\| = 1$, the rate at which $f$ increases along $\mathbf{u}$ is

$$
D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \|\nabla f\|\cos\theta
$$

This is the **directional derivative** (proved by linearizing $f$ along the step; chapter 06 supplies the approximation). Since $|\cos\theta| \le 1$, the maximum is at $\theta = 0$: $\mathbf{u} = \dfrac{\nabla f}{\|\nabla f\|}$, with rate $\|\nabla f\|$. Negative gradient → steepest *decrease*.

**Worked example.** At $(1,2)$, $\nabla f = (8, 3)$, $\|\nabla f\| = \sqrt{64+9} = \sqrt{73} \approx 8.54$.
- Fastest ascent direction: $(8, 3)/\sqrt{73} \approx (0.937, 0.351)$ — rate $\sqrt{73} \approx 8.54$.
- Fastest descent: the opposite direction, $(-0.937, -0.351)$.
- Along $(1,0)$ (pure $x$-direction): rate $= (8,3)\cdot(1,0) = 8$.
- Along $(-3,8)$ (orthogonal to the gradient: $(8,3)\cdot(-3,8) = -24+24 = 0$): rate $= 0$ — to first order $f$ does not change; this direction runs along a contour.

**2. Zero gradient at a smooth extremum.** If $f$ has a local min or max at an interior point, $\nabla f = \mathbf{0}$ — every partial vanishes. Optimizing means (approximately) solving $\nabla f = \mathbf{0}$; gradient descent approaches that equation iteratively instead of solving it directly.

## Connection: Gradient Descent (Preview)

Because $-\nabla f$ points in the steepest descent direction:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta_{\text{old}})
$$

with learning rate $\eta > 0$. Section 05 derives this carefully (step size, convergence, stochastic variants) — the formula's *direction* comes from this chapter, its *step size* from chapter 06, and its *behavior* from section 05.

## Connection to AI

- **The loss is a function of all parameters.** $J(\theta_1, \ldots, \theta_d)$ with $d$ possibly billions; $\nabla J$ has one component per parameter. "Computing gradients" in training means computing this one vector.
- **Steepest descent intuition** is literal: for small steps, no direction reduces the loss faster than $-\nabla J$ (normalized). Descent methods search along it.
- **Gradient norm as a status signal.** $\|\nabla J\| \approx 0$ means a (possibly flat) critical point; a large norm means training still has room. Monitoring it is standard practice.
- **Per-example gradients.** The total loss $J = \frac{1}{n}\sum_i L_i$ has $\nabla J = \frac{1}{n}\sum_i \nabla L_i$ (linearity of differentiation) — the fact that lets stochastic gradient descent use a subset (section 05).

## Common Mistakes

- Differentiating with respect to the wrong variable: with $y$ fixed, $(3xy)_x = 3y$, **not** $3y$ times something else — treat other variables as ordinary constants.
- Forgetting to transpose: gradients are column vectors; writing $\nabla f = (8, 3)$ is fine informally, but in equations $\theta - \eta\nabla J$ requires consistent shapes.
- Confusing $\nabla f$ (a vector field, defined at every point) with a single evaluated vector (one number tuple).
- Assuming $\nabla f = \mathbf{0}$ guarantees an extremum — saddle points also have zero gradient (chapter 07).
- Treating $\|\nabla f\|$ as the directional derivative in every direction — it is the rate only along the gradient itself.

## Exercises

1. Compute $\dfrac{\partial f}{\partial x}$ and $\dfrac{\partial f}{\partial y}$ for $f(x,y) = x^2 y + y^3$.
2. Evaluate both partials of the same $f$ at $(2, 1)$.
3. Find $\nabla f$ for $f(x, y) = e^{x+y}$ at $(0, 0)$, and its length.
4. The directional derivative of $f$ along a unit $\mathbf{u}$ is $\nabla f \cdot \mathbf{u}$. At a point where $\nabla f = (4, 0)$, compute the rate of change along $\mathbf{u} = (0, 1)$ and interpret geometrically.
5. For $L(w, b) = (wx + b - y)^2$ with numbers $x = 1$, $y = 3$: find $\dfrac{\partial L}{\partial w}$ and $\dfrac{\partial L}{\partial b}$ at $w = 2$, $b = 0$.
6. State the first-order condition for an interior local extremum of a differentiable $f(x, y)$.
