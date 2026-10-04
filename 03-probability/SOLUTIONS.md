# 03 — Probability: Solutions

## 01-basics.md

**1. Card from a 52-card deck.**
$P(\text{ace}) = \frac{4}{52} = \frac{1}{13}$.
$P(\text{heart}) = \frac{13}{52} = \frac{1}{4}$.
$P(\text{ace or heart}) = P(A) + P(H) - P(A \cap H) = \frac{4}{52} + \frac{13}{52} - \frac{1}{52} = \frac{16}{52} = \frac{4}{13}$
(the intersection is the ace of hearts — one card).

**2. Two coins.**
$\Omega = \{HH, HT, TH, TT\}$, each with probability $1/4$.
$P(\text{both heads}) = P(\{HH\}) = 1/4$.
$P(\text{at least one head}) = 1 - P(TT) = 1 - 1/4 = 3/4$.

**3. $P(A \cup B)$ with $P(A)=0.4$, $P(B)=0.5$, $P(A \cap B)=0.2$.**
$0.4 + 0.5 - 0.2 = 0.7$.

**4. Die: $A$ = even, $B$ = $>3$.**
$A \cap B = \{4, 6\}$, so $P(A \cap B) = 2/6 = 1/3$.
Addition rule: $P(A \cup B) = \frac12 + \frac12 - \frac13 = \frac23$.
Direct count: $A \cup B = \{2,4,5,6\}$, $4/6 = 2/3$ ✓

**5. $P(A^c) = 1 - P(A)$.**
$A$ and $A^c$ are disjoint and $A \cup A^c = \Omega$, so by additivity and normalization:
$P(A) + P(A^c) = P(\Omega) = 1 \Rightarrow P(A^c) = 1 - P(A)$.

**6. Why $P(\varnothing) = 0$ is forced.**
$\varnothing$ is disjoint from $\Omega$ and $\varnothing \cup \Omega = \Omega$, so additivity gives $P(\varnothing) + P(\Omega) = P(\Omega)$, i.e., $P(\varnothing) + 1 = 1$, so $P(\varnothing) = 0$. (Non-negativity then confirms consistency: $0 \ge 0$.)

## 02-conditional-and-bayes.md

**1. $P(A \mid B)$ with $P(A \cap B) = 0.12$, $P(B) = 0.3$.**
$\frac{0.12}{0.3} = 0.4$.

**2. $P(\text{sum} = 7 \mid \text{first die even})$.**
First die even: $18$ equally likely pairs. Sum 7 with first even: $(2,5), (4,3), (6,1)$ → $3$ pairs.
$$
P = \frac{3}{18} = \frac{1}{6}
$$

**3. Verify the multiplication rule.**
$P(\text{sum}=7 \cap \text{first even}) = 3/36 = 1/12$. $P(\text{first even}) = 18/36 = 1/2$.
Conditional: $\frac{1/12}{1/2} = \frac{1}{6}$ ✓ matches problem 2.

**4. Two balls without replacement, 3 red + 2 blue: $P(\text{both red})$.**
$$
P(\text{both red}) = P(\text{2nd red} \mid \text{1st red})\,P(\text{1st red}) = \frac{2}{4}\cdot\frac{3}{5} = \frac{6}{20} = \frac{3}{10}
$$
(First draw: 3 of 5 red; given that, 2 of the remaining 4 are red.)

**5. Bayes with $P(B \mid A) = 0.8$, $P(A) = 0.25$, $P(B) = 0.4$.**
$$
P(A \mid B) = \frac{0.8 \times 0.25}{0.4} = \frac{0.2}{0.4} = 0.5
$$

**6. Medical test, prevalence $0.001$.**
$$
P(+ \mid D)P(D) = 0.99 \times 0.001 = 0.00099
$$
$$
P(+) = 0.00099 + 0.05 \times 0.999 = 0.00099 + 0.04995 = 0.05094
$$
$$
P(D \mid +) = \frac{0.00099}{0.05094} \approx 0.0194 \approx 1.9\%
$$
Lower than the $16.7\%$ from prevalence $1\%$ — with the *same* test quality, a rarer disease yields a smaller posterior. The prior directly scales the answer; likelihood alone never determines it.

**7. Mutually exclusive events with positive probabilities are not independent.**
Die: $A = \{1\}$, $B = \{2\}$: disjoint, so $P(A \cap B) = 0$, but $P(A)P(B) = \frac16 \cdot \frac16 = \frac{1}{36} \ne 0$.
$P(A \cap B) \ne P(A)P(B)$ → **not independent**. Intuitively: knowing the roll was 1 tells you *certainly* that it was not 2 — maximal dependence, not none.

## 03-random-variables.md

**1. PMF of the sum of two dice, $x = 2 \ldots 7$.**

| $x$ | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| $p(x)$ | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 |

(counts: $(1,1)$; $(1,2)(2,1)$; $(1,3)(2,2)(3,1)$; …; six pairs for 7.)
Total for 2–7: $\frac{1+2+3+4+5+6}{36} = \frac{21}{36} = \frac{7}{12}$.
(For the full PMF over 2–12, the counts run 1,2,3,4,5,6,5,4,3,2,1 — summing to 36/36 = 1 ✓)

**2. Bernoulli($p = 0.6$).**
$p(1) = 0.6$, $p(0) = 0.4$; all other values: $0$. Sums to 1 ✓

**3. Uniform on $[0,4]$.**
$f(x) = \frac{1}{4-0} = 0.25$ for $0 \le x \le 4$, else $0$.
$$
P(1 \le X \le 2) = \int_1^2 0.25\,dx = 0.25 \times 1 = 0.25
$$

**4. $X \sim N(0,1)$, $P(-1 \le X \le 1)$.**
$\mu \pm \sigma = 0 \pm 1$: about **68%** by the 68–95–99.7 rule.

**5. Discrete or continuous?**
(a) number of errors — **discrete** (count);
(b) predicted probability of a class — **continuous** (a real value in $[0,1]$; it is produced by a continuous function of the inputs);
(c) token index — **discrete** (categorical);
(d) model weight — **continuous** (a real parameter).

**6. $F(5) = 0.7$, $F(3) = 0.4$.**
$$
P(3 < X \le 5) = F(5) - F(3) = 0.7 - 0.4 = 0.3
$$

## 04-expectation-variance.md

**1. $\mathbb{E}[X]$ for $p = (0.2, 0.5, 0.3)$ on $\{1,2,3\}$.**
$1(0.2) + 2(0.5) + 3(0.3) = 0.2 + 1.0 + 0.9 = 2.1$.

**2. $\operatorname{Var}(X)$.**
$$
\mathbb{E}[X^2] = 1(0.2) + 4(0.5) + 9(0.3) = 0.2 + 2.0 + 2.7 = 4.9
$$
$$
\operatorname{Var}(X) = 4.9 - (2.1)^2 = 4.9 - 4.41 = 0.49, \qquad \sigma = 0.7
$$

**3. Bernoulli($p = 0.7$).**
$\mathbb{E}[X] = 0.7$; $\operatorname{Var}(X) = p(1-p) = 0.7 \times 0.3 = 0.21$.

**4. $\operatorname{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$.**
$11 - (2)(5) = 11 - 10 = 1$ (positive association).

**5. Variances and correlation for the joint table.**
$\mathbb{E}[X^2] = 0^2(0.5) + 1^2(0.5) = 0.5$; $\operatorname{Var}(X) = 0.5 - 0.25 = 0.25$; same by symmetry for $Y$.
$\sigma_X = \sigma_Y = 0.5$, $\operatorname{Cov} = 0.05$ (from the chapter):
$$
\rho = \frac{0.05}{0.5 \times 0.5} = \frac{0.05}{0.25} = 0.2
$$
Weak positive correlation.

**6. $\operatorname{Cov}(aX + b, Y) = a\operatorname{Cov}(X,Y)$.**
Mean of $aX + b$ is $a\mu_X + b$, so
$$
\mathbb{E}[(aX + b - (a\mu_X + b))(Y - \mu_Y)]
= \mathbb{E}[a(X - \mu_X)(Y - \mu_Y)]
= a\,\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
= a\operatorname{Cov}(X,Y)
$$
using linearity to pull out the constant $a$ (and $b$ cancels in the centering).

**7. Independent $X, Y$ ⇒ $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$.**
$$
\mathbb{E}[XY] = \sum_x \sum_y x\,y\,p(x,y)
\overset{\text{indep}}{=} \sum_x \sum_y x\,y\,p(x)p(y)
= \left(\sum_x x\,p(x)\right)\left(\sum_y y\,p(y)\right)
= \mathbb{E}[X]\,\mathbb{E}[Y]
$$
(The step marked "indep" is exactly where the assumption enters; without it the joint does not split.)

## 05-joint-marginal-conditional.md

**1. Marginals.**
$p(X{=}0) = 0.3 + 0.2 = 0.5$; $p(Y{=}0) = 0.3 + 0.2 = 0.5$.

**2. Conditionals and independence.**
$$
p(Y{=}0 \mid X{=}0) = \frac{0.3}{0.5} = 0.6, \qquad
p(Y{=}0 \mid X{=}1) = \frac{0.2}{0.5} = 0.4
$$
The conditional distribution of $Y$ changes with $X$ ($0.6 \ne 0.4$) → **not independent**.
(Equivalently: $p(0,0) = 0.3 \ne 0.5 \times 0.5 = 0.25$.)

**3. Verify Bayes.**
$$
p(X{=}0 \mid Y{=}1) = \frac{p(X{=}0, Y{=}1)}{p(Y{=}1)} = \frac{0.2}{0.5} = 0.4
$$
$$
\frac{p(X{=}0 \mid Y{=}1)\,p(Y{=}1)}{p(X{=}0)} = \frac{0.4 \times 0.5}{0.5} = 0.4 = p(Y{=}1 \mid X{=}0) \;✓
$$

**4. Two independent fair coins.**
$p(x_1, x_2) = p(x_1)p(x_2) = \frac12 \cdot \frac12 = \frac14$ for each of the four pairs.
$X_1 + X_2 = 2$ only at $(1,1)$: probability $1/4$.

**5. Chain rule for three variables.**
$$
p(x_1, x_2, x_3) = p(x_1)\,p(x_2 \mid x_1)\,p(x_3 \mid x_1, x_2)
$$

**6. Sequence probability.**
$$
p(w_1, w_2, w_3) = 0.5 \times 0.4 \times 0.3 = 0.06
$$
(one factor from the chain rule per token).

**7. Conditional independence statement.**
With $X$ = yesterday's price, $Z$ = today's price, $Y$ = tomorrow's price:
$$
Y \perp X \mid Z
$$
"Given today's price, yesterday's price and tomorrow's price are independent" — a Markov assumption: the present screens off the past from the future.
