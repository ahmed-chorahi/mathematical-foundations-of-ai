# Conditional Entropy

## Why It Matters for AI

The whole point of a model is that *something known reduces uncertainty*: knowing the pixels narrows the label, knowing the prefix narrows the next token. **Conditional entropy** $H(Y \mid X)$ is the precise amount of uncertainty about $Y$ that remains *given* $X$ — the irreducible error of the best possible predictor that sees $X$. Comparing $H(Y)$ with $H(Y\mid X)$ measures how much knowledge is worth; that comparison is the next chapter's mutual information.

## Definition

$H(Y \mid X) = \mathbb{E}_{x \sim p(x)}\big[\,H(Y \mid X = x)\,\big]$ — average, over values of $X$, of the entropy of $Y$ restricted to that slice:

$$
\boxed{\;H(Y \mid X) = -\sum_{x} p(x) \sum_y p(y \mid x) \log p(y \mid x)
= -\sum_{x,y} p(x, y) \log p(y \mid x)\;}
$$

The outer expectation over $x$, the inner entropy of the conditional distribution $p(\cdot \mid x)$.

**Worked example — word choice given previous word.** Let $Y$ be the next word from $\{$`a`, `b`$\}$ and $X$ the previous one, with joint probabilities:

| | $Y=$`a` | $Y=$`b` | $p(x)$ |
|---|---|---|---|
| $X=$`u` | 0.4 | 0.1 | 0.5 |
| $X=$`v` | 0.1 | 0.4 | 0.5 |

Column/row marginals: $p(X{=}\texttt{u}) = p(X{=}\texttt{v}) = 0.5$; $p(Y{=}\texttt{a}) = p(Y{=}\texttt{b}) = 0.5$.

Slices: $p(Y \mid \texttt{u}) = [0.8, 0.2]$, $p(Y \mid \texttt{v}) = [0.2, 0.8]$.

$$
H(Y \mid X{=}\texttt{u}) = H([0.8, 0.2]) = -(0.8\ln 0.8 + 0.2\ln 0.2) \approx 0.5004 \text{ nats}
$$

identically for $\texttt{v}$; weight by $p(x) = \tfrac12$ each:

$$
H(Y \mid X) = \tfrac12(0.5004) + \tfrac12(0.5004) = 0.5004 \text{ nats}
$$

Compare the marginal $H(Y) = H([0.5, 0.5]) = \ln 2 \approx 0.6931$ nats: knowing $X$ removed $0.6931 - 0.5004 = 0.1927$ nats of uncertainty — the next chapter's quantity (mutual information).

**Independent case:** if $X, Y$ independent, $p(y \mid x) = p(y)$ for every $x$, so

$$
H(Y \mid X) = \sum_x p(x) H(Y) = H(Y)
$$

— learning $X$ tells you nothing; conditional entropy equals unconditional entropy.

**Deterministic case:** if $Y$ is a function of $X$ (label given pixels, in the noiseless idealization), each slice is a point mass: $H(Y \mid X = x) = 0$ for all $x$, hence $H(Y \mid X) = 0$ — perfect prediction.

## Joint and Chain Rules

**Joint entropy** of the pair:

$$
H(X, Y) = -\sum_{x,y} p(x,y) \log p(x, y)
$$

**Chain rule for entropy** (always true):

$$
\boxed{\;H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)\;}
$$

*Reason:* expand $H(X,Y) = -\sum p(x,y)\log p(x,y)$ and substitute $p(x,y) = p(x)\,p(y \mid x)$; the sum splits into one part that is $H(X)$ and one that is $H(Y \mid X)$ (probability's own chain rule — section 03 — inside a logarithm).

Reading: the joint uncertainty = uncertainty of the first part + whatever remains of the second given the first. In the worked example:

$$
H(X, Y) = H(X) + H(Y \mid X) = 0.6931 + 0.5004 = 1.1935 \text{ nats}
$$

(direct check: $-0.4\ln0.4 - 0.1\ln0.1 - 0.1\ln0.1 - 0.4\ln0.4 = 0.3665 + 0.2303 + 0.2303 + 0.3665 = 1.1936$ ✓ within rounding).

## Properties and Relations

1. **Range:** $0 \le H(Y \mid X) \le H(Y)$ — conditioning cannot create uncertainty *on average* (**conditioning reduces entropy**, $H(Y \mid X) \le H(Y)$), though for *specific* values $x$ the slice entropy can exceed $H(Y)$.
   *Proof sketch of the inequality:* $H(Y) - H(Y \mid X) = I(X; Y) \ge 0$ (chapter 05) or via non-negativity of conditional KL — we prove it there.
2. **Symmetry in the chain:** $H(X,Y) = H(Y,X)$ implies $H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$ — two ways to peel the joint.
3. **Data processing intuition:** functions of knowledge can only reduce remaining uncertainty: if $Z = f(X)$, then $H(Y \mid Z) \ge H(Y \mid X)$ (more information, less residual — stated as a fact).
4. **Continuous analog** exists ($h(Y \mid X)$ with densities) — not used here.

## Connection to AI

- **Bayes error as conditional entropy.** If labels are genuinely random given inputs (ambiguous images, noisy annotators), $H(Y \mid X) > 0$ is the uncertainty no classifier can remove: the best achievable log-loss *is* $H(Y\mid X)$. Train/test cross-entropy below this would be impossible without leakage; above it is the model's avoidable error. This is section 04's "irreducible error", now measured exactly.
- **Language modeling:** $H(W_t \mid W_1 \ldots W_{t-1})$ — the conditional entropy of the next token given the prefix — is *the* number a language model drives down (perplexity $= \exp$ of it, section 10). Context reduces uncertainty; the reduction is quantified here.
- **Features are valuable iff $H(Y \mid X_{\text{new}}) < H(Y \mid X_{\text{old}})$** — the criterion behind feature selection and "is this additional input worth it?".
- **Chain rule = autoregressive factorization:** $H(X_1, \ldots, X_T) = \sum_t H(X_t \mid X_{<t})$ — the mathematical license for models that predict one token at a time (section 10 uses this constantly).
- **Decision trees** split on variables that reduce conditional entropy most — the literal algorithmic use of this chapter's function.

## Common Mistakes

- Reading $H(Y \mid X)$ as a single number per $x$ — it is the *average* over $x$; $H(Y \mid X = x)$ varies by slice (the table shows two slices, which happened to be equal).
- Assuming conditioning always strictly helps — independence gives equality; some $x$-slices can even be more uncertain than marginal (average still cannot exceed $H(Y)$).
- Mixing $H(Y \mid X)$ with $H(Y, X)$ — remaining vs total uncertainty; chain rule connects them, they are not equal.
- Dropping the outer expectation: writing $-\log p(y \mid x)$ without weighting by $p(x)$ gives a per-slice number, not the conditional entropy.
- Forgetting that $H(Y\mid X) = 0$ requires a *function* relationship in the true distribution — a model that memorized training labels does not make $H(Y \mid X)$ small on new data.

## Exercises

1. Compute $H(Y \mid X)$ for: $X \in \{1, 2\}$ uniform; $Y \mid X{=}1$ deterministic at `a`; $Y \mid X{=}2$ deterministic at `a` as well. What does the answer say about $X$'s usefulness?
2. $X$ uniform on 2 values; $H(Y \mid X = 1) = 1$ nat, $H(Y \mid X = 2) = 3$ nats, both slices equally likely. Compute $H(Y \mid X)$.
3. If $Y = X$ exactly (binary, uniform): compute $H(Y \mid X)$ and $H(X, Y)$.
4. Chain rule check with the chapter's table: compute $H(Y) + H(X \mid Y)$ and confirm it equals $1.1935$ nats. ($H(X \mid Y) = H(X, Y) - H(Y)$.)
5. State the independent case: what is $H(Y \mid X)$ when $X \perp Y$, in one line?
6. A model's test cross-entropy is $0.60$ nats; the data's $H(Y \mid X) = 0.55$ nats. What does the $0.05$ gap represent? What would a score of $0.50$ signal?
