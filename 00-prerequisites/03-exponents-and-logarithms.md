# Exponents and Logarithms

## Why It Matters for AI

Probability multiplies, and information is logarithmic. Softmax is built from $e^x$; entropy, cross-entropy, and log-likelihood are built from $\log$. Gradient descent often manipulates products by taking logarithms first. This chapter carries the identities used constantly from section 03 onward — learn them here so later sections never stop to re-explain them.

## Exponents

For a positive base $a$ and integers:

$$
a^n = \underbrace{a \cdot a \cdots a}_{n \text{ times}} \qquad
a^{-n} = \frac{1}{a^n} \qquad
a^0 = 1 \ (a \neq 0)
$$

The laws, valid for all real exponents when $a > 0$:

| law | identity | example |
|---|---|---|
| product | $a^m a^n = a^{m+n}$ | $2^3 \cdot 2^4 = 2^7$ |
| quotient | $\dfrac{a^m}{a^n} = a^{m-n}$ | $\dfrac{5^6}{5^2} = 5^4$ |
| power | $(a^m)^n = a^{mn}$ | $(3^2)^3 = 3^6$ |
| root | $a^{1/n} = \sqrt[n]{a}$ | $8^{1/3} = 2$ |

**Worked example.** Simplify $\dfrac{2^{n+1}}{2^{n-1}} = 2^{(n+1)-(n-1)} = 2^{2} = 4$. The $n$ cancels — a pattern that shows up when ratios of exponentials are normalized in softmax.

## The Number $e$

$e \approx 2.718281828\ldots$ is defined by the limit

$$
e = \lim_{n \to \infty}\left(1 + \frac{1}{n}\right)^n
=\; 1 + 1 + \frac{1}{2!} + \frac{1}{3!} + \cdots
$$

Why this particular constant: it is the base for which the exponential function equals its own derivative.

$$
\frac{d}{dx}e^x = e^x
$$

That single property makes $e^x$ the natural language of growth, decay, and — later — of gradients: the functions whose rates equal their values. When $e$ appears in a formula it is usually a choice of convenience, not a physical necessity; changing the base only rescales logarithms by a constant factor.

The general exponential $a^x$ is defined through $e$:

$$
a^x = e^{x \ln a}
$$

where $\ln a$ is the natural logarithm of $a$.

## Logarithms

$\log_a x$ answers the question: "to what power must $a$ be raised to produce $x$?"

$$
y = \log_a x \iff a^y = x \qquad (a > 0,\ a \neq 1,\ x > 0)
$$

Two bases dominate:

- $\log_2 x$ — **binary logarithm**, measured in **bits**. Used in information theory (section 06): the number of yes/no questions needed to pin down an outcome.
- $\ln x = \log_e x$ — **natural logarithm**, the default in mathematics and machine learning. Written $\ln$ or just $\log$ in later sections; the base only multiplies results by a constant.

### Identities

These are the ones to memorize:

$$
\log_a(xy) = \log_a x + \log_a y
$$

$$
\log_a\left(\frac{x}{y}\right) = \log_a x - \log_a y
$$

$$
\log_a(x^n) = n\log_a x
$$

$$
\log_a a = 1, \qquad \log_a 1 = 0
$$

**Change of base** (convert between bases):

$$
\log_a x = \frac{\log_b x}{\log_b a}
$$

The product rule is the entire reason logarithms exist in practice: **products become sums**. The likelihood of many independent observations is a product of probabilities; taking its logarithm turns the product into a sum, which is easier to differentiate and to reason about.

**Worked example.** Expand $\log_2(8x^3)$:

$$
\log_2(8x^3) = \log_2 8 + \log_2 x^3 = 3 + 3\log_2 x
$$

**Worked example.** Solve $2^{3x} = 16$.

Write both sides with base 2: $2^{3x} = 2^4$, so $3x = 4$ and $x = 4/3$. Alternatively take $\log_2$ of both sides — valid because the logarithm is a one-to-one function.

## Inverse Relationship

$$
a^{\log_a x} = x \qquad (x > 0)
$$

$$
\log_a(a^y) = y
$$

So $\ln(e^x) = x$ and $e^{\ln x} = x$ for $x > 0$. Each undoes the other — exactly the inverse-function structure of section 02.

## Useful Limits and Approximations

$$
\lim_{x \to 0} \frac{e^x - 1}{x} = 1, \qquad \lim_{x \to 0}\frac{\ln(1+x)}{x} = 1
$$

For small $x$: $e^x \approx 1 + x$ and $\ln(1+x) \approx x$. These are the first-order Taylor approximations you will derive properly in section 02; they are mentioned now because derivations will use them.

Logarithm growth: $\log x$ grows without bound but ever more slowly — $\log_{10}$ reaches 6 only at $10^6$. This compresses wide ranges: probability $10^{-6}$ becomes $-6$ in log scale, $10^{-3}$ becomes $-3$.

## Connection to AI

- **Softmax uses $e^x$:** $\operatorname{softmax}(z_i) = e^{z_i}/\sum_j e^{z_j}$. Positivity of $e^{x}$ guarantees a valid probability; the normalizing denominator enforces $\sum_i = 1$ (section 09).
- **Log-likelihood turns products into sums:** with independent data, $P(\text{all}) = \prod_t p_t$, so $\log P = \sum_t \log p_t$. Maximizing the sum maximizes the product (section 04).
- **Cross-entropy and entropy are logarithmic:** $H = -\sum p \log p$ (section 06). The logarithm makes rare, surprising events cost more — the "surprise" of an outcome is $-\log p$.
- **Temperature divides logits:** $\operatorname{softmax}(z/T)$ — scaling before the exponential, directly from these identities (section 10).
- **Bits:** $\log_2$ measures information in bits, the unit of entropy (section 06).

## Common Mistakes

- $\log(x + y) \ne \log x + \log y$. Only multiplication splits into a sum.
- $\log_a x$ is undefined for $x \le 0$. Writing $\ln 0 = -\infty$ names a limit, not a number usable in arithmetic.
- Confusing $a^x$ with $x^a$: $2^3 = 8$ but $3^2 = 9$.
- Assuming $\log_2 x$ and $\ln x$ give the same number: $\ln x = (\ln 2)\log_2 x \approx 0.693 \log_2 x$.
- Dropping the absolute value when integrating $1/x$ later: $\int \frac{1}{x}dx = \ln|x| + C$.

## Exercises

1. Simplify: $\dfrac{3^{2n}}{3^{n-1}}$.
2. Evaluate: $\log_2 32$, $\log_{10} 1000$, $\ln 1$.
3. Expand $\ln\left(\dfrac{x^2 y}{z^3}\right)$ as a sum of logs.
4. Solve for $t$: $e^{0.5t} = 4$ (give an exact form and a decimal approximation to two places; $\ln 4 \approx 1.386$).
5. Show that $\log_a x \cdot \log_x a = 1$ for valid $a, x$.
6. A coin has probability $p = 0.01$ of heads. Compute $-\log_2 p$ and state what the number means as a count of yes/no questions.
