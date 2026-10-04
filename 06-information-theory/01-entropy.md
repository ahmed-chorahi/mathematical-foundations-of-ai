# Entropy

## Why It Matters for AI

Before you can reduce uncertainty — predict a token, classify an image — you need a number that *measures* it. **Entropy** is that number: the average surprise of a distribution, in nats (or bits). It sets the floor for prediction (you cannot beat the entropy of the true distribution with certainty), it measures what a model should be learning to represent, and it gives the units in which every later quantity in this section is expressed.

## Surprise: The Self-Information of an Event

How surprising is it that an event with probability $p$ happened? It must be:

- **zero** when $p = 1$ (no surprise — it had to happen),
- **increasing** as $p \to 0$ (rarer events are more surprising),
- **additive** for independent events (two independent surprises add — a coin flip and a die roll are not "jointly as surprising as one").

The unique function (up to a positive scale) with these properties:

$$
I(x) = -\log p(x) \qquad (\text{nats})
$$

Base $e$: nats. Base 2: **bits** — $I(x) = -\log_2 p(x)$, and a fair coin flip gives $-\log_2 \tfrac12 = 1$ bit, recovering the everyday unit.

Examples (natural log, verified): $p = 1/2 \Rightarrow I = \ln 2 \approx 0.693$ nats; $p = 0.01 \Rightarrow I = \ln 100 \approx 4.605$ nats; $p = 0.001 \Rightarrow I = \ln 1000 \approx 6.908$ nats — an event one tenth as likely is $\ln 10 \approx 2.303$ nats *more* surprising (the difference of two self-informations is a log-ratio). Averaging $I$ over a distribution weights rare events heavily — the fact that shapes the definition next.

## Definition: Entropy

**Entropy** is the *expected* surprise under the distribution itself:

$$
\boxed{\;H(p) = \mathbb{E}_{x \sim p}\big[-\log p(x)\big] = -\sum_{i} p_i \log p_i\;}
$$

**Worked example — fair coin.**

$$
H = -\left(\tfrac12\ln\tfrac12 + \tfrac12\ln\tfrac12\right) = -\ln\tfrac12 = \ln 2 \approx 0.693 \text{ nats} = 1 \text{ bit}
$$

**Worked example — biased die** $p = [0.7, 0.2, 0.1]$:

$$
H = -(0.7\ln 0.7 + 0.2\ln 0.2 + 0.1\ln 0.1)
$$

$$
0.7\ln 0.7 \approx -0.2497,\quad 0.2\ln 0.2 \approx -0.3219,\quad 0.1\ln 0.1 \approx -0.2303
$$

$$
H \approx 0.8018 \text{ nats} \approx 1.157 \text{ bits}
$$

(also $0.8018 / 0.6931 \approx 1.157$ — the bits/nats conversion by $\ln 2$ ✓).

```python
from math import log
def entropy(p):
    return -sum(pi * log(pi) for pi in p if pi > 0)
print(entropy([0.5, 0.5]))       # 0.6931 nats
print(entropy([0.7, 0.2, 0.1]))  # 0.8018 nats
```

(The `if pi > 0` handles the limit $0\log 0 = 0$ — a distribution may assign zero probability; such outcomes contribute no expected surprise because they never occur.)

## Properties

1. **Non-negative:** $H(p) \ge 0$, with $H = 0$ iff the distribution is deterministic ($p = [1, 0, \dots]$) — no uncertainty, no surprise.
2. **Maximum at uniform:** over $k$ outcomes, $H \le \ln k$, equality iff $p_i = 1/k$. (Proof sketch: the maximum of $\sum p_i \ln(1/p_i)$ is Jensen's-inequality-at-the-maximum, or Lagrange multipliers on $\sum p_i = 1$; either way: uniform wins.) A fair 6-sided die: $\ln 6 \approx 1.792$ nats.
3. **Continuous in the distribution:** small changes in $p$ change $H$ smoothly — entropy can be tracked during training.
4. **Additivity for independent variables:** $H(X, Y) = H(X) + H(Y)$ when $X, Y$ independent — two fair coins: $2\ln 2$ nats = 2 bits (the joint outcome is genuinely twice as uncertain).
5. **Units are the log's base:** nats (ln), bits ($\log_2$); converting: bits $=$ nats $/\ln 2$.

## The Coding Meaning (Why This Function)

Shannon's source-coding theorem: to transmit outcomes of a distribution $p$, the best **average** code length (per symbol, in bits, with codes allowed to differ in length — short codes for likely symbols) equals the entropy $H_2(p)$ exactly. A fair coin needs 1 bit per flip on average; the biased $[0.7, 0.2, 0.1]$ source needs $1.157$ bits (as computed) — no scheme does better in the long run.

We do not prove the theorem; the takeaway: **entropy is not an arbitrary formula — it is the optimal price of communicating uncertainty.** That theorem also justifies $-\log p$ as the right cost of a single event (self-information = ideal code length), and hence averages to $H$.

## Connection to AI

- **Entropy of a prediction** measures a model's *self-reported* uncertainty: a softmax output with $H \approx 0$ claims to be certain; $H$ near $\ln k$ claims to be guessing. Monitoring training entropy shows when the model stops learning.
- **Irreducible error:** if the true label distribution has entropy $H(p) > 0$ (ambiguous examples), no classifier gets certainty — some test error is the data's own entropy, not the model's fault. The Bayes-optimal *expected* log-loss equals $H(p)$ (next chapter makes this exact).
- **Perplexity** (section 10) is $\exp(H)$ in nats — a language model's entropy re-expressed as "equally likely choices", the standard LLM quality number.
- **Regularization & pruning** literature uses entropy to describe weight or output distributions (entropy of attention, etc. — sections 08–09).
- **Temperature** (section 10) reshapes a distribution to *change* its entropy: raising temperature flattens toward uniform, increasing $H$ — sampling diversity is entropy control.

## Common Mistakes

- Mixing bases: $0.693$ vs $1$ for the same coin — state your base; training code usually uses nats (natural log) silently.
- Forgetting $0 \log 0 = 0$ — without it, a zero-probability entry breaks the formula; take the limit.
- Reading entropy as "how uniform $p$ is" — it measures the probability *skew* over outcomes: any two-outcome distribution $(q, 1-q)$ has $H$ determined by $q$ alone, independent of what the outcomes are called.
- Assuming more entropy is better or worse — it is a measurement; a confident-correct model has low output entropy, and that is good only when correctness accompanies it.
- Computing $-\sum p \log q$ and calling it entropy (that's cross-entropy — chapter 02).

## Exercises

1. Compute $H(p)$ in nats for $p = [0.5, 0.5]$, $p = [0.9, 0.1]$, $p = [0.25, 0.25, 0.25, 0.25]$.
2. Which has higher entropy, $[0.9, 0.1]$ or $[0.6, 0.4]$? Give a one-sentence intuition (skew).
3. Express 2 nats in bits; 3 bits in nats.
4. Fair 8-sided die: entropy in bits — and confirm $\ln 8$ in nats via conversion.
5. Why is $H(p) = 0$ only for deterministic distributions? (One sentence with the definition.)
6. Two independent fair coins: use additivity to get the joint entropy $H(X, Y)$ in bits, then verify by enumerating the 4 joint outcomes directly.
