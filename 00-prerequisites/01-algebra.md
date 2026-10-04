# Algebra

## Why It Matters for AI

Every equation in AI is an equation to be manipulated. A gradient is set to zero and solved for the optimum; a loss is rewritten to make a term visible; an inequality defines what "better" means. Algebra is the skill of rearranging symbolic statements without losing or inventing information. Later sections never re-teach it — this chapter makes sure the foundation holds.

## Numbers and Variables

A **variable** is a symbol standing for a number we may not know yet. We write $x$ for an unknown input, $\theta$ for a model parameter, $n$ for a count. An **expression** like $3x + 2$ is a recipe for producing a number once $x$ is given. An **equation** like $3x + 2 = 11$ asserts that two expressions name the same number; it is a statement that can be true or false depending on $x$.

Conventions used throughout this repository:

- lowercase letters ($x, y, w$) are numbers or scalars,
- Greek letters ($\theta, \eta, \lambda$) are parameters or rates,
- a subscript ($x_1, x_2$) labels a member of a list rather than a power.

## Solving Equations

Solving $3x + 2 = 11$ means isolating $x$ while keeping both sides equal. The rule is: whatever you do to one side, do to the other.

$$
3x + 2 = 11
\;\;\xrightarrow{\text{subtract } 2}\;\;
3x = 9
\;\;\xrightarrow{\text{divide by } 3}\;\;
x = 3
$$

**Check the answer** by substituting back: $3(3) + 2 = 11$. ✓

Two operations appear constantly in AI derivations:

**Isolate a variable inside a product.** From $ab = c$ with $b \neq 0$:

$$
a = \frac{c}{b}
$$

The condition $b \neq 0$ matters — division by zero is undefined, and it will show up again when a covariance matrix is singular.

**Isolate a variable inside a square.** From $x^2 = 9$:

$$
x = 3 \quad \text{or} \quad x = -3
$$

Squares lose sign information; recovering it introduces both roots. In optimization, $x^2 = c$ often yields $x = \pm\sqrt{c}$, and choosing the right root is a modeling decision.

**Rearranging a formula.** In statistics you will meet

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

and need to solve for $n$ in a specific case. Multiply both sides by $n - 1$, divide by $s^2$, add 1. The order of operations is reversed: undo addition first, then multiplication, then powers.

## Linear and Quadratic Equations

A **linear equation** in one unknown has the form $ax + b = 0$ with $a \neq 0$; it has exactly one solution $x = -b/a$.

A **quadratic equation** $ax^2 + bx + c = 0$ has the quadratic formula:

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

The term $b^2 - 4ac$ (the **discriminant**) decides the number of real solutions:

| discriminant | solutions |
|---|---|
| $> 0$ | two distinct real solutions |
| $= 0$ | one repeated real solution |
| $< 0$ | no real solutions (two complex ones) |

This appears directly in linear algebra: the eigenvalues of a matrix are roots of a polynomial, and the discrimininant determines whether they are real.

**Worked example.** Solve $x^2 - 5x + 6 = 0$.

$a = 1,\; b = -5,\; c = 6$. Discriminant: $(-5)^2 - 4(1)(6) = 25 - 24 = 1 > 0$, so two solutions:

$$
x = \frac{5 \pm 1}{2} \;\Rightarrow\; x = 3 \;\text{ or }\; x = 2
$$

Check: $3^2 - 5(3) + 6 = 9 - 15 + 6 = 0$ ✓ and $2^2 - 5(2) + 6 = 4 - 10 + 6 = 0$ ✓.

Factoring finds the same answer faster when it works: $x^2 - 5x + 6 = (x-2)(x-3)$, and a product is zero exactly when a factor is zero.

## Inequalities

An **inequality** compares magnitudes: $a < b$, $a \le b$, $a \ge b$. They behave like equations with two exceptions:

1. **Adding or subtracting** the same quantity preserves the direction: $a < b \Rightarrow a + c < b + c$.
2. **Multiplying or dividing by a negative number reverses** the direction: $a < b \Rightarrow -a > -b$.

Example: solve $-2x + 1 > 5$.

$$
-2x > 4 \;\xrightarrow{\text{divide by } -2}\; x < -2
$$

Intervals are written $(a, b)$ for $a < x < b$ (endpoints excluded), $[a, b]$ for $a \le x \le b$ (included), and $\infty$ / $-\infty$ for unbounded ends — $(0, \infty)$ means all positive numbers.

Why AI cares: optimization states targets as inequalities — "reduce the loss *below* 0.1", "keep $\|\theta\|$ *small*" — and constraints are inequalities. Probability rests on $0 \le p(x) \le 1$.

## Absolute Value

$$
|x| = \begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}
$$

$|x|$ is the distance from $x$ to $0$ on the number line, and $|x - y|$ is the distance between $x$ and $y$. This is why absolute value defines an error size that ignores direction: $|x - 3| < 0.01$ means "within 0.01 of 3".

To solve $|x - 3| = 5$, split into both cases: $x - 3 = 5$ or $x - 3 = -5$, giving $x = 8$ or $x = -2$.

## Connection to AI

- **Model training solves equations.** Setting a gradient to zero, $\nabla J(\theta) = 0$, produces equations whose solutions are candidate optima.
- **Inequalities describe constraints and metrics.** An accuracy of at least 95% is $A \ge 0.95$; a constrained optimization problem (section 05) is a set of inequalities plus an objective.
- **Rearrangement is derivation work.** Every formula you will see for a gradient, a posterior, or a learning-rate schedule gets to its final form by repeated legal rearrangement.
- **The $\pm$ habit matters.** Solutions to squared equations have two branches; in practice one is rejected because a quantity (a probability, a variance) cannot be negative.

## Common Mistakes

- Dividing by a variable that might be zero. Always state the condition ($b \neq 0$ above) or consider the zero case separately.
- Forgetting to reverse the inequality when multiplying by a negative number.
- Writing $x^2 = 9 \Rightarrow x = 3$ and dropping the negative root.
- Confusing $2x$ (two times $x$) with $x^2$ (x squared).

## Exercises

1. Solve for $x$: $4x - 7 = 2x + 5$.
2. Solve for $w$: $\frac{w}{3} + 2 = 8$.
3. Solve $x^2 - 2x - 8 = 0$ by factoring, and verify both roots.
4. Solve the inequality $5 - 3x \le 2$.
5. Rearrange $E = \frac{1}{2}mv^2$ to make $v$ the subject, assuming $m > 0$ and $v \ge 0$.
6. Solve $|2x + 1| = 7$.
