# Derivatives

## Why It Matters for AI

The derivative answers one question precisely: *if I nudge the input a little, how much does the output change?* Training a model means nudging parameters to reduce a loss — and without derivatives you cannot know which nudge helps. This chapter builds the tool; sections 03 and 05 turn it into the gradient and backpropagation.

## Definition

The **derivative** of $f$ at $a$ is the limit of the slope between $a$ and a nearby point $h$ away:

$$
f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}
$$

$\dfrac{f(a+h)-f(a)}{h}$ is the **difference quotient**: average rate of change over the interval of width $h$. Sending $h \to 0$ shrinks the interval until, if the limit exists, average becomes instantaneous.

Equivalent notations: $f'(x)$, $\dfrac{df}{dx}$, $\dfrac{d}{dx}f(x)$.

**Geometric reading:** $f'(a)$ is the slope of the tangent line at $(a, f(a))$.

**Worked example — derivative from the definition.** $f(x) = x^2$:

$$
\frac{(x+h)^2 - x^2}{h} = \frac{x^2 + 2xh + h^2 - x^2}{h} = 2x + h
\;\xrightarrow{h \to 0}\;
f'(x) = 2x
$$

**Interpretation.** At $x = 3$: $f'(3) = 6$ — near $3$, each tiny increase in $x$ multiplies the increase in $x^2$ by about $6$. If $x$ grows by $0.01$, $x^2$ grows by about $0.06$.

## Derivative Rules

Each rule exists to avoid re-taking limits.

| rule | statement | example |
|---|---|---|
| constant | $(c)' = 0$ | a flat function never changes |
| power | $(x^n)' = n x^{n-1}$ | $(x^5)' = 5x^4$ |
| constant multiple | $(c\,f)' = c f'$ | $(4x^3)' = 12x^2$ |
| sum | $(f + g)' = f' + g'$ | differentiate term by term |
| product | $(fg)' = f'g + fg'$ | $(x e^x)' = e^x + x e^x$ |
| quotient | $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$ | $\left(\dfrac{1}{u}\right)' = -\dfrac{u'}{u^2}$ |

The **chain rule** for composed functions is substantial enough for its own chapter (05).

**Worked example — product rule.** $f(x) = x^2 \sin x$:

$$
f'(x) = 2x\sin x + x^2\cos x
$$

($f'g + fg'$ with $f = x^2$, $g = \sin x$.)

**Worked example — quotient rule.** $g(x) = \dfrac{x}{x^2+1}$:

$$
g'(x) = \frac{(1)(x^2+1) - (x)(2x)}{(x^2+1)^2} = \frac{1 - x^2}{(x^2+1)^2}
$$

## Common Derivatives

$$
(e^x)' = e^x
\qquad
(\ln x)' = \frac{1}{x}
\qquad
(\sin x)' = \cos x
\qquad
(\cos x)' = -\sin x
$$

Combined with the power rule, these cover most of what appears in AI:

$$
\frac{d}{dx}\ln(x^2+1) = \frac{1}{x^2+1}\cdot 2x = \frac{2x}{x^2+1}
$$

(that last step used the chain rule applied to $\ln u$ with $u = x^2+1$ — the structure $\frac{d}{dx}\ln u = \frac{u'}{u}$ recurs in log-likelihoods.)

## A Derivative That Appears Everywhere: the Sigmoid

$\sigma(x) = \dfrac{1}{1+e^{-x}} = (1 + e^{-x})^{-1}$. Apply the power rule plus chain rule:

$$
\sigma'(x) = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}
$$

Now use $\sigma(x) = \frac{1}{1+e^{-x}}$ and $1 - \sigma(x) = \frac{e^{-x}}{1+e^{-x}}$ (verified in section 00):

$$
\frac{e^{-x}}{(1+e^{-x})^2}
= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}}
= \sigma(x)\,(1 - \sigma(x))
$$

$$
\boxed{\;\sigma'(x) = \sigma(x)(1-\sigma(x))\;}
$$

Why this form matters:

- **Cheap to compute** once $\sigma(x)$ is known — no extra exponentials.
- **Maximum at $x = 0$**: $\sigma'(0) = \tfrac12 \cdot \tfrac12 = \tfrac14$. The sigmoid is flattest as $|x|$ grows ($\sigma' \to 0$) — gradients vanish for large-magnitude inputs, the origin of the *vanishing gradient* problem for saturated units.
- Range: $0 < \sigma' \le 1/4$ — the sigmoid can never amplify a gradient.

## Higher Derivatives

The derivative of the derivative:

$$
f''(x) = \frac{d}{dx}f'(x), \qquad f'''(x), \ldots
$$

$f''$ measures **curvature**: concave up ($f'' > 0$, slope increasing), concave down ($f'' < 0$), flat ($f'' = 0$). For $f(x) = x^2$: $f'(x) = 2x$, $f''(x) = 2$ — constant upward curvature.

Second derivatives of the loss (as a Hessian — chapter 04) decide whether a critical point is a minimum (section 07).

## Linear Approximation

For small $h$, differentiability gives:

$$
f(a + h) \approx f(a) + f'(a)h
$$

The tangent line replaces the function locally. This is the approximation that gradient descent exploits (chapter 06 develops it properly); for now, a numeric check:

$f(x) = x^2$ at $a = 3$, $h = 0.1$: approximation $9 + 6(0.1) = 9.6$; exact $3.1^2 = 9.61$. Error $0.01$ — second order in $h$.

## Connection to AI

- **Loss sensitivity.** $J'(\theta) = -2$ at the current point says: increasing $\theta$ by $0.01$ decreases the loss by about $0.02$. Training is entirely this computation for every parameter.
- **Parameter updates** in every model are $\theta \leftarrow \theta - \eta J'(\theta)$ (section 05) — a derivative evaluated at the current $\theta$, then applied.
- **Activation derivatives** are baked into backpropagation: the network's backward pass multiplies activation derivatives layer by layer (section 08). $\sigma' = \sigma(1-\sigma)$ appears in logistic regression's gradient derivation (section 07).
- **Gradient of log-likelihood** — the $u'/u$ pattern of $\ln'$ produces the familiar "prediction minus target" gradient of cross-entropy (sections 07, 08).

## Common Mistakes

- $(fg)' \ne f'g'$ — the product rule has two terms.
- $(f/g)' \ne f'/g'$ — same mistake for quotients.
- Forgetting the inner derivative: $(e^{3x})' = 3e^{3x}$, not $e^{3x}$.
- Dropping negative signs with $\ln$ or $\cos$: $(\cos x)' = -\sin x$, $(\ln x)' = 1/x > 0$ for $x > 0$ but $\ln(x^{-1})' $ requires care.
- Assuming $f'' = 0$ means an extremum — $f(x) = x^3$ at $0$ has zero second derivative and no extremum (section 07).

## Exercises

1. Differentiate $f(x) = 4x^3 - 2x + 7$.
2. Compute $f'(2)$ for $f(x) = x^2$ directly from the limit definition.
3. Differentiate $g(x) = (2x+1)^5$. (Structure: power rule outside, linear inner function — practice the chain-rule shape even before chapter 05.)
4. Differentiate $h(x) = x e^{x}$ using the product rule.
5. Using the limit definition or the power rule, show that $\frac{d}{dx}\frac{1}{x} = -\frac{1}{x^2}$.
6. Evaluate $\sigma'(0)$ for the sigmoid and interpret: what fraction of an input change passes into the output at $x = 0$? Compare with $\sigma'(5)$ ($\sigma(5) \approx 0.993$).
7. For $f(x) = \ln(x^2 + 1)$, find $f'(x)$ and $f'(0)$.
