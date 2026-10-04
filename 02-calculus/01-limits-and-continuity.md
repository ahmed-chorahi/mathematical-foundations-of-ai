# Limits and Continuity

## Why It Matters for AI

The derivative is defined as a limit, and a function must behave well enough for that limit to exist. This chapter states precisely what "approaches" and "no jumps" mean — enough to justify every derivative rule that follows, without a formal analysis course. It also explains why ReLU's corner at zero is harmless and why smooth activations are convenient.

## The Limit

$$
\lim_{x \to a} f(x) = L
$$

reads: "as $x$ gets arbitrarily close to $a$ (without reaching it), $f(x)$ gets arbitrarily close to $L$."

Two clarifications:

- **The value at $a$ does not matter.** A limit studies behavior *near* $a$, not *at* $a$. The function may be undefined at $a$; the limit can still exist.
- **Approaching from both sides.** The limit exists only when the left-hand limit ($x \to a^-$) and right-hand limit ($x \to a^+$) agree.

**Worked example 1.** $f(x) = \dfrac{x^2 - 1}{x - 1}$ at $x \to 1$:

Direct substitution gives $0/0$ — undefined. Factor instead:

$$
\frac{x^2-1}{x-1} = \frac{(x-1)(x+1)}{x-1} = x + 1 \quad (x \ne 1)
$$

So $\lim_{x\to1} f(x) = 2$. The function is undefined at exactly $x = 1$, yet the limit exists — behavior near $a$ versus at $a$.

**Worked example 2 — a limit that does not exist.** The function $g(x) = 0$ for $x < 0$ and $g(x) = 1$ for $x \ge 0$: approaching $0$ from the left gives $0$, from the right gives $1$. Disagreement → no limit at $0$ — a jump.

### Limit laws

If $\lim f(x) = L$ and $\lim g(x) = M$, then:

$$
\lim(f + g) = L + M, \qquad
\lim(f \cdot g) = LM, \qquad
\lim\frac{f}{g} = \frac{L}{M}\ (M \ne 0)
$$

and $\lim (c\,f) = cL$. These are what "plugging in" actually uses — they are valid **only when the parts have limits** (the $0/0$ case above fails the division law, which is why factoring was needed).

For continuous functions at the evaluation point, the practical rule is: **substitute**.

## Continuity

$f$ is **continuous** at $a$ if:

1. $f(a)$ is defined,
2. $\lim_{x\to a} f(x)$ exists,
3. $\lim_{x\to a} f(x) = f(a)$.

Informally: the graph has no hole, jump, or break at $a$. A function continuous at every point in its domain is simply called continuous.

Examples:

| function | continuous? | where |
|---|---|---|
| polynomials $x^2 - 4x + 5$ | yes | everywhere |
| $\dfrac{1}{x}$ | yes | everywhere except $x = 0$ (not defined there) |
| $\sqrt{x}$ | yes | on $[0, \infty)$; at $0$ the right-hand limit equals $f(0) = 0$ |
| $\text{ReLU}(x) = \max(0,x)$ | yes | everywhere — both pieces give $0$ at $x = 0$ |
| sign function $\operatorname{sgn}(x)$ | no | jumps at $0$ |

**Composition and sums** of continuous functions are continuous — so $e^{-x}$, $\ln(x^2+1)$, and $\frac{1}{1+e^{-x}}$ (composition of continuous pieces) are continuous wherever defined. Softmax is therefore continuous.

## Differentiability (What Comes Next)

The derivative will be defined as a limit of difference quotients. Two facts connect continuity to what follows:

- **Differentiable ⇒ continuous.** If $f'(a)$ exists, $f$ is continuous at $a$. (The converse is false.)
- **Continuity does not guarantee a derivative.** ReLU at $x = 0$ is continuous but has a corner: the left slope is $0$, the right slope is $1$; no single tangent exists.

Why this matters in practice: at a corner the derivative is undefined *at that one point* — a set of measure zero. Training algorithms assign some value there (frameworks conventionally use $0$ for ReLU's derivative at $0$) and continue. Functions that are continuous but not differentiable at a few points cause no practical trouble; functions that jump *discontinuously* (like the sign function) break gradient-based methods outright because no slope information exists at all.

## Connection to AI

- **Activation functions are chosen for their smoothness.** Sigmoid, tanh, and softmax are differentiable everywhere; ReLU is not differentiable at $0$ yet works because the exceptional set is negligible. Knowing exactly what continuity and differentiability mean lets you evaluate such claims instead of taking them on faith.
- **Loss functions must be differentiable** for gradient descent to apply — that is a statement about limits.
- **Numerical care.** $\log$ and $1/x$ have singularities; implementations guard against them ($\log(x + \varepsilon)$, clipped denominators) precisely because the limit fails there.

## Common Mistakes

- Assuming $\lim_{x\to a} f(x) = f(a)$ always — it is the *definition* of continuity, to be checked, not assumed.
- Canceling a factor without noting the excluded point: $\frac{x^2-1}{x-1} = x+1$ only for $x \ne 1$.
- Confusing "no limit" with "infinite limit": $1/x \to \infty$ as $x \to 0^+$ describes a one-sided tendency; the two-sided limit at $0$ does not exist as a real number.
- Believing differentiable functions need continuous derivatives — $f(x) = x^2\sin(1/x)$ (with $f(0)=0$) is differentiable everywhere but its derivative oscillates near $0$.

## Exercises

1. Evaluate $\lim_{x \to 3} (x^2 + 2x)$ and justify with the limit laws (or continuity).
2. Compute $\lim_{x \to 2} \dfrac{x^2 - 4}{x - 2}$ by factoring.
3. Does $\lim_{x \to 0} \dfrac{|x|}{x}$ exist? Evaluate from both sides.
4. Is $f(x) = \dfrac{1}{x-1}$ continuous at $x = 1$? Which condition fails?
5. Is ReLU continuous at $0$? Check the three conditions of continuity.
6. Give an example of a function with a limit existing at a point where the function itself is undefined, and state the value.
