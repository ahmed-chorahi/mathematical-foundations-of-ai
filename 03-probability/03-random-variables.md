# Random Variables and Distributions

## Why It Matters for AI

Outcomes become numbers — class indices, scores, measurements — and probability theory moves from sets of outcomes to functions on the real line. A **random variable** is that bridge; its **distribution** (PMF, PDF, or CDF) is the complete description of its behavior. Softmax outputs are discrete distributions; Gaussian assumptions are continuous; labels are categorical. This chapter defines the objects later chapters compute with.

## Random Variables

A **random variable** $X$ is a numerical function on the sample space $\Omega$:

$$
X: \Omega \to \mathbb{R}
$$

- Die roll → $X(\omega) = \omega$ (the number showing).
- Two coin flips → $X$ = number of heads, values $\{0, 1, 2\}$.
- Image classification → $X$ = predicted class index $\{1, \ldots, K\}$.

The random variable *converts* outcomes into numbers so we can take averages, plot distributions, and compute losses. Two random variables can share a distribution without being the same function.

**Discrete** random variable: countable set of values $\{x_1, x_2, \ldots\}$. **Continuous**: any value in an interval (uncountably many — chapter-style: real-valued measurements).

## Discrete Distributions: the PMF

The **probability mass function** $p(x) = P(X = x)$ assigns each possible value its probability:

- $p(x) \ge 0$ for all $x$,
- $\sum_x p(x) = 1$ (all mass accounted for).

**Worked example — two coin flips.** $X$ = number of heads, $\Omega = \{HH, HT, TH, TT\}$ equally likely:

| $x$ | 0 | 1 | 2 |
|---|---|---|---|
| $p(x)$ | 1/4 | 2/4 | 1/4 |

$P(X = 1) = 2/4$ because $HT$ and $TH$ both give one head. Sum: $1/4 + 2/4 + 1/4 = 1$ ✓

### Common discrete distributions

**Bernoulli($p$)** — one trial, success/failure: $p(1) = p$, $p(0) = 1-p$.
AI: binary labels, dropout masks, a single token's correct/incorrect.

**Categorical / Multinomial-1 ($p_1, \ldots, p_K$)** — one draw among $K$ classes: $P(X = k) = p_k$, $\sum p_k = 1$.
AI: class labels; *the softmax output is a categorical distribution over the vocabulary* — the central object of section 10.

**Uniform on $\{1, \ldots, N\}$**: $p(k) = 1/N$ — every outcome equally likely.

**Bernoulli worked example.** $p = 0.3$ (say, probability a user clicks): $p(1) = 0.3$, $p(0) = 0.7$. Mean of many such trials → about $30\%$ successes (chapter 04 makes this expectation).

## Continuous Durations: PDF and CDF

For a continuous $X$, every single value has probability $0$ (infinitely many candidates share total mass 1) — so probabilities are assigned to *intervals* via a **probability density function** $f(x)$:

$$
P(a \le X \le b) = \int_a^b f(x)\,dx
$$

with $f(x) \ge 0$ and $\int_{-\infty}^{\infty} f(x)\,dx = 1$.

$f(x)$ is a *density*, not a probability: $f$ may exceed 1 (e.g., uniform on $[0, 0.1]$ has $f = 10$); only the area under $f$ is a probability. $P(X = a) = \int_a^a f = 0$.

### The CDF

For both kinds of random variable:

$$
F(x) = P(X \le x)
$$

- Discrete: $F(x) = \sum_{t \le x} p(t)$ — a staircase.
- Continuous: $F(x) = \int_{-\infty}^{x} f(t)\,dt$ — smooth, with $F' = f$ where differentiable.

$F$ is non-decreasing, $F(-\infty) = 0$, $F(\infty) = 1$; $P(a < X \le b) = F(b) - F(a)$.

### Common continuous distributions

**Uniform on $[a, b]$:** $f(x) = \dfrac{1}{b-a}$ on the interval — constant density; $P$ of any subinterval = its length / total length.

**Gaussian (normal) $N(\mu, \sigma^2)$:**

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- $\mu$ = center (mean), $\sigma^2$ = spread (variance).
- The bell curve: $\mu \pm \sigma$ holds $\approx 68\%$, $\mu \pm 2\sigma$ $\approx 95\%$, $\mu \pm 3\sigma$ $\approx 99.7\%$ of the mass.
- Standard normal $N(0,1)$; any $N(\mu,\sigma^2)$ is $\mu + \sigma Z$ for $Z \sim N(0,1)$.

Why the Gaussian dominates: sums of many independent small effects concentrate into a Gaussian shape (central limit theorem — stated, not proved), and it is the maximum-entropy distribution given fixed mean and variance (section 06 makes entropy precise). AI uses it for noise injection, initialization assumptions, and as a simple posterior family.

**Exponential** — waiting times, decay; not needed here but appears in arrival models.

## Which to Choose

| situation | type | example |
|---|---|---|
| categories, counts | discrete PMF | class label, token id, number of clicks |
| measurements, errors | continuous PDF | pixel values, loss values, embeddings' coordinates |
| probability of "at most" something | CDF | $P(\text{score} \le 0.9)$ |

## Connection to AI

- **A classifier's output is a PMF.** Softmax over $K$ classes produces $p_1, \ldots, p_K$ with $p_k \ge 0$ and $\sum = 1$ — a categorical distribution. Evaluating it on the true label is $p_{\text{true}}$ — the quantity cross-entropy penalizes (section 06).
- **Labels are random variables** during training: $Y \in \{1, \ldots, K\}$ with an unknown distribution $P(Y)$ — often imbalanced (different $p_k$), which is the prior from chapter 02.
- **Continuous assumptions.** Gaussian noise, Gaussian-weight initialization, and pixel likelihood models (e.g., image generation with Gaussian output noise) are modeling choices — wrong in detail, useful in practice.
- **The chain of section 10** — tokens as categorical draws conditioned on context — starts with exactly the definition given here.

## Common Mistakes

- Treating a PDF value as a probability: $f(0.5) = 3$ is legal; probabilities are areas.
- Forgetting $\sum p(x) = 1$ when constructing a PMF (a common counting slip).
- Writing $P(X = x)$ for a continuous random variable as anything but $0$.
- Confusing $p(x)$ (the distribution of $X$) with $P(X \le x)$ (the CDF).
- Assuming the Gaussian distribution models anything and everything — it is a modeling assumption; heavy-tailed data violate it.

## Exercises

1. Two fair dice: let $X$ = sum. List the PMF of $X$ for values 2 through 7 and verify the listed masses sum to at least the total for that range.
2. Give the PMF of a Bernoulli random variable with $p = 0.6$.
3. A continuous uniform $X$ on $[0, 4]$: give the PDF and compute $P(1 \le X \le 2)$.
4. For $X \sim N(0, 1)$, state $P(-1 \le X \le 1)$ approximately, using the 68–95–99.7 rule.
5. Classify each as discrete or continuous: (a) number of errors in a batch; (b) predicted probability of a class; (c) a token index; (d) a weight in a trained model.
6. The CDF of a random variable satisfies $F(5) = 0.7$, $F(3) = 0.4$. Give $P(3 < X \le 5)$.
