# Summation and Notation

## Why It Matters for AI

A training loss is a sum over data points. An expectation is a weighted sum. Entropy is a sum over outcomes. Reading $\sum$ notation comfortably is a precondition for every formula in statistics, information theory, and machine learning — and this chapter also fixes the set notation and Greek letters used throughout the repository.

## Summation Notation

$$
\sum_{i=1}^{n} x_i \;=\; x_1 + x_2 + \cdots + x_n
$$

Read it as: "sum $x_i$ over $i$ from 1 to $n$". The symbol underneath is the **index** (or dummy variable) of summation; it runs through every integer from the lower limit to the upper limit. The index name is arbitrary:

$$
\sum_{i=1}^{3} x_i \;=\; \sum_{k=1}^{3} x_k \;=\; x_1 + x_2 + x_3
$$

### Rules

| rule | identity |
|---|---|
| constant factors out | $\sum_{i=1}^n c\,x_i = c\sum_{i=1}^n x_i$ |
| sums add | $\sum_{i=1}^n (x_i + y_i) = \sum_{i=1}^n x_i + \sum_{i=1}^n y_i$ |
| split a range | $\sum_{i=1}^{n} x_i = \sum_{i=1}^{m} x_i + \sum_{i=m+1}^{n} x_i$ |

### Common closed forms

$$
\sum_{i=1}^{n} c = cn
$$

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

**Worked example.** Compute $\sum_{i=1}^{4} (2i + 1)$:

$$
2\sum_{i=1}^4 i + \sum_{i=1}^4 1 = 2\cdot\frac{4\cdot5}{2} + 4 = 20 + 4 = 24
$$

Check by brute force: $(2\cdot1{+}1)+(2\cdot2{+}1)+(2\cdot3{+}1)+(2\cdot4{+}1) = 3+5+7+9 = 24$ ✓

### Nested sums

A sum inside a sum iterates the inner index for each outer index:

$$
\sum_{i=1}^{2}\sum_{j=1}^{3} i\,j \;=\; (1\cdot1 + 1\cdot2 + 1\cdot3) + (2\cdot1 + 2\cdot2 + 2\cdot3) = 6 + 12 = 18
$$

Matrix multiplication is a nested sum (section 01): $(AB)_{ik} = \sum_j A_{ij}B_{jk}$.

### Infinite sums

$\sum_{i=0}^{\infty} x^i = \dfrac{1}{1-x}$ for $|x| < 1$. This geometric series generates the sigmoid-like expansions used in proofs; for now, note the condition — an infinite sum only "converges" (approaches a finite value) when its terms shrink fast enough.

## Products

$$
\prod_{i=1}^{n} x_i = x_1 x_2 \cdots x_n
$$

Likelihoods are products of probabilities; the logarithm converts them to sums. This pairing (product in probability → sum in log-space) recurs throughout sections 03–06.

## Set Notation

Probability is built on sets. The vocabulary:

- **Set**: a collection of distinct elements. $A = \{1, 2, 3\}$.
- **Element**: $2 \in A$ reads "2 is in A". $5 \notin A$.
- **Union**: $A \cup B$ — elements in $A$ *or* $B$ (or both).
- **Intersection**: $A \cap B$ — elements in *both*.
- **Complement**: $A^c$ (or $\bar A$) — elements not in $A$.
- **Empty set**: $\varnothing$ — contains nothing.
- **Subset**: $A \subseteq B$ — every element of $A$ is in $B$.
- **Universal set**: $S$ (or $\Omega$ in probability) — everything under discussion.

Worked example: $A = \{1,2,3\}$, $B = \{3,4\}$.

$$
A \cup B = \{1,2,3,4\}, \quad A \cap B = \{3\}, \quad A \setminus B = \{1,2\}
$$

De Morgan's laws (used when negating conditions):

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

"not (A or B)" = "not A and not B". Check with the example: $(A\cup B)^c$ removes $\{1,2,3,4\}$; $A^c \cap B^c$ keeps only things outside both — same result.

**Counting finite sets.** If $A$ and $B$ are disjoint (no overlap):

$$
|A \cup B| = |A| + |B|
$$

In general the **inclusion–exclusion** rule:

$$
|A \cup B| = |A| + |B| - |A \cap B|
$$

This is precisely the addition rule of probability in section 03.

## Functions Between Sets

A function $f: A \to B$ (section 02) is fully specified by its **domain** $A$, **codomain** $B$, and rule. In probability, a random variable is a function from outcomes to numbers — this is why knowing the function and the distribution of the input determines the distribution of the output.

## Greek Letters Used in AI

| letter | name | common use |
|---|---|---|
| $\theta$ | theta | model parameters, collected into one vector |
| $\eta$ | eta | learning rate |
| $\lambda$ | lambda | regularization strength, eigenvalue (usually $\mu$ or $\lambda$) |
| $\mu$ | mu | mean |
| $\sigma$ | sigma | standard deviation; $\sigma^2$ variance |
| $\alpha$ | alpha | learning rate variants, significance level |
| $\beta$ | beta | momentum coefficient, distribution parameters |
| $\nabla$ | nabla | gradient operator |
| $\epsilon$ | epsilon | small quantity |
| $\delta$ | delta | error term in backpropagation |
| $\Sigma$ | capital sigma | covariance matrix; also the summation sign (context distinguishes) |
| $\omega, \Omega$ | omega | outcomes, sample space; big-$\Omega$ growth bound |
| $\chi^2$ | chi-squared | test statistic |

Also: $\mathbb{E}$ for expectation, $P$ for probability, $\operatorname{Var}$, $\operatorname{Cov}$, $\mathbb{R}^n$ for the set of real $n$-dimensional vectors.

## Connection to AI

- **Loss is a sum:** $J = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i)$ — an average over $n$ training examples, written in one line with $\sum$.
- **Expectation is a weighted sum:** $\mathbb{E}[X] = \sum_x x\,p(x)$ (section 03).
- **Entropy is a sum over outcomes:** $H = -\sum_x p(x)\log p(x)$ (section 06).
- **Sets are events:** $A \cap B$ = "both events happened", the core of conditional probability (section 03).
- **The loss *gradient* is also a sum:** $\nabla J = \frac{1}{n}\sum_i \nabla L_i$ — which is why stochastic gradient descent, using one term instead of all, is an approximation (section 05).

## Common Mistakes

- Misreading bounds: $\sum_{i=0}^{n} x_i$ has $n+1$ terms, not $n$.
- Pulling a *variable* out of a sum: $\sum c x_i = c\sum x_i$ only when $c$ does not involve $i$.
- Treating $\sum_{i=1}^n (x_i + y_i)$ as $\left(\sum x_i\right)\left(\sum y_i\right)$ — products of sums expand differently.
- Confusing $A \cup B$ with "A and B": union is *or*, intersection is *and*.
- Writing $\sum$ with no upper limit and assuming it converges.

## Exercises

1. Evaluate $\sum_{i=1}^{5} i^2$ by listing the terms.
2. Simplify $\sum_{i=1}^{n} (3i - 2)$ using the closed forms above. (Answer form: an expression in $n$.)
3. Given $A = \{a, b, c\}$, $B = \{b, c, d\}$, compute $A \cup B$, $A \cap B$, and $A \setminus B$.
4. Verify De Morgan's law for $A = \{1,2\}$, $B = \{2,3\}$ with universal set $S = \{1,2,3,4\}$.
5. Evaluate $\sum_{i=1}^{3}\sum_{j=1}^{2} (i + j)$.
6. Write "the average of the squared errors of $n$ predictions $\hat{y}_i$ against targets $y_i$" as a formula using $\sum$.
