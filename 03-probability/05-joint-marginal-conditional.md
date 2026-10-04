# Joint, Marginal, and Conditional Distributions

## Why It Matters for AI

Real problems involve several random variables at once — features and label, current token and context, parameters and data. The **joint distribution** holds all information about how variables co-occur; **marginals** and **conditionals** are what you get by summing it out or renormalizing a slice of it. Language modeling is literally the factorization of a joint distribution into conditionals — this chapter supplies the machinery.

## Joint Distribution

For two discrete random variables, the **joint PMF**:

$$
p(x, y) = P(X = x, \; Y = y)
$$

a table (or grid) of probabilities over all pairs, satisfying $p(x,y) \ge 0$ and $\sum_{x,y} p(x,y) = 1$.

**Worked example — a small joint table.** $X \in \{0,1\}$ (feature), $Y \in \{0,1\}$ (label):

| $p(x,y)$ | $Y=0$ | $Y=1$ | row total |
|---|---|---|---|
| $X=0$ | 0.3 | 0.2 | **0.5** |
| $X=1$ | 0.2 | 0.3 | **0.5** |
| col total | **0.5** | **0.5** | 1 |

Continuous case: a **joint density** $f(x,y)$ with $P((X,Y) \in R) = \iint_R f$ — same rules, integrals instead of sums.

The joint contains everything: any question about $X$ and $Y$ together or separately can be answered from it.

## Marginal Distributions

Drop a variable by summing it out:

$$
p(x) = \sum_y p(x, y)
$$

(the name comes from writing totals in the *margins* of the table). Continuous: $f_X(x) = \int f(x,y)\,dy$.

From the table: $p(X{=}0) = 0.3 + 0.2 = 0.5$, $p(X{=}1) = 0.2 + 0.3 = 0.5$; similarly $p(Y{=}0) = p(Y{=}1) = 0.5$ — the bold totals.

Marginalization **discards information**: knowing the marginals $p(x)$ and $p(y)$ does not determine the joint (many tables share those margins — the table above, and one with $0.4, 0.1, 0.1, 0.4$, both have uniform margins). Dependence lives in the joint.

## Conditional Distributions

Divide the joint by the marginal of the conditioning variable:

$$
\boxed{\;p(y \mid x) = \frac{p(x, y)}{p(x)}\;}
\qquad (p(x) > 0)
$$

This is chapter 02's definition, now for distributions: for each fixed $x$, $p(\cdot \mid x)$ is a valid PMF over $y$ (non-negative; sums to $\sum_y p(x,y)/p(x) = p(x)/p(x) = 1$).

**Worked example — from the table:**

$$
p(Y{=}1 \mid X{=}0) = \frac{p(X{=}0, Y{=}1)}{p(X{=}0)} = \frac{0.2}{0.5} = 0.4
$$

$$
p(Y{=}1 \mid X{=}1) = \frac{0.3}{0.5} = 0.6
$$

Knowing $X$ changes the distribution of $Y$ ($0.4$ vs $0.6$) — the variables are dependent, and *this* is the object a classifier learns: $p(Y \mid X)$, the label distribution given the features.

**Bayes for distributions** follows the same algebra as chapter 02:

$$
p(y \mid x) = \frac{p(x \mid y)\,p(y)}{p(x)}
$$

## Independence of Random Variables

$X$ and $Y$ are **independent** iff the joint factors everywhere:

$$
p(x, y) = p(x)\,p(y) \quad \text{for all } x, y
$$

equivalently $p(y \mid x) = p(y)$ for all $x$ with $p(x) > 0$ — conditioning on $X$ never changes $Y$'s distribution.

Check the example: $p(0,0) = 0.3$ but $p(X{=}0)\,p(Y{=}0) = 0.25$ — not equal → **dependent**.

Independence of *many* variables: the joint factorizes completely $p(x_1, \ldots, x_n) = \prod_i p(x_i)$ — the strongest assumption available; real data rarely satisfies it, which is why models condition on context.

## Chain Rule for Distributions

The multiplication rule generalizes to any number of variables — factor the joint into conditionals:

$$
\boxed{\;p(x_1, \ldots, x_n) = p(x_1)\,p(x_2 \mid x_1)\,p(x_3 \mid x_1, x_2) \cdots p(x_n \mid x_1, \ldots, x_{n-1})\;}
$$

*Derivation:* iterate $p(a, b) = p(a)\,p(b \mid a)$:

$$
p(x_1, \ldots, x_n) = p(x_1)\,p(x_2, \ldots, x_n \mid x_1)
$$

then apply the same rule inside the conditional (valid because a conditional distribution is itself a probability distribution, so all its rules apply):

$$
p(x_2, \ldots, x_n \mid x_1) = p(x_2 \mid x_1)\,p(x_3, \ldots, x_n \mid x_1, x_2)
$$

and continue. Equality holds **always** — no independence assumption. Independence would collapse it to the product of marginals.

**This is how language models work.** For a sequence of tokens $w_1, \ldots, w_T$:

$$
p(w_1, \ldots, w_T) = \prod_{t=1}^{T} p(w_t \mid w_1, \ldots, w_{t-1})
$$

Predicting "the next token given everything before" *is* one factor of this chain. Section 10 builds on it; nothing beyond this identity is needed to see why next-token prediction determines a distribution over whole sequences.

## Conditional Independence

$X \perp Y \mid Z$ means: given $Z$, learning $Y$ tells you nothing about $X$:

$$
p(x, y \mid z) = p(x \mid z)\,p(y \mid z)
$$

(The joint *conditioned on $Z$* factors; unconditionally $X$ and $Y$ may be strongly related through $Z$ — two symptoms both caused by one disease are dependent marginally, independent given the disease.)

Why it matters: conditional independence assumptions are how tractable models are built — naive Bayes (features independent given the class), hidden Markov models (observations independent given states), and attention masks (a token's representation built only from permitted context) are all statements of this kind.

## Connection to AI

- **$p(y \mid x)$ is the prediction task.** Classification models approximate the conditional label distribution; everything in training estimates it from data.
- **Sequence models = chain rule.** Next-token factorization above is the definition of what an LLM models (section 10).
- **Marginalization is integration over what you ignore.** Computing $p(y) = \sum_x p(y \mid x)p(x)$ appears when a latent variable is summed out (mixture models, VAEs' evidence bound — both reduce to this arithmetic).
- **Generative vs discriminative:** modeling the joint $p(x, y)$ (then deriving $p(y \mid x)$ via Bayes) versus modeling $p(y \mid x)$ directly — two strategies defined by this chapter's objects.
- **Bayes' theorem with distributions** gives MAP/MLE (section 04).

## Common Mistakes

- Summing the joint over the *wrong* variable: $p(x) = \sum_y p(x,y)$ — to get $X$'s marginal, remove $Y$.
- Forgetting to renormalize: a slice $p(x, y_0)$ is *not* a valid PMF in $x$ until divided by $p(y_0)$ — that division is exactly the conditional.
- Assuming independence from equal marginals — margins say nothing about the joint (two tables, same margins).
- Misreading the chain rule as requiring independence — it always holds; independence would *simplify* it to a product of marginals.
- Confusing $p(x \mid y)$ (distribution over $X$ for fixed $y$) with $p(y \mid x)$ — Bayes translates between them, they are not interchangeable.

## Exercises

1. Marginalize the chapter's joint table: compute $p(X{=}0)$, $p(Y{=}0)$.
2. Using the same table, compute $p(Y{=}0 \mid X{=}0)$ and $p(Y{=}0 \mid X{=}1)$; are $X$ and $Y$ independent?
3. For the table, verify Bayes: check that $p(Y{=}1 \mid X{=}0) = \frac{p(X{=}0 \mid Y{=}1)\,p(Y{=}1)}{p(X{=}0)}$ by computing $p(X{=}0 \mid Y{=}1)$ first.
4. Two independent fair coins $X_1, X_2$ (heads = 1): write the joint PMF $p(x_1, x_2)$ as a product of marginals and compute $p(X_1 + X_2 = 2)$.
5. Factor $p(x_1, x_2, x_3)$ with the chain rule (write the three factors explicitly).
6. A language model over 3 tokens has $p(w_1) = 0.5$; $p(w_2 = a \mid w_1) = 0.4$; $p(w_3 = b \mid w_1, w_2) = 0.3$. Give the probability of that particular sequence.
7. Give a conditional-independence statement describing "given today's price, yesterday's price does not help predict tomorrow's price" — name the three variables.
