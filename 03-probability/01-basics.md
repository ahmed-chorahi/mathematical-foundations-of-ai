# Probability Basics

## Why It Matters for AI

A model trained on data must make statements about events it has not observed: *this image contains a cat*, *the next token is "the"*. Probability is the framework that makes such statements precise and quantitatively consistent. Everything later — conditional probability, distributions, likelihood, cross-entropy — rests on the axioms and counting rules of this chapter.

## Sample Spaces and Events

A **sample space** $\Omega$ is the set of all possible outcomes of an experiment.

- Roll of a die: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Coin flips until heads: $\Omega = \{H, TH, TTH, \ldots\}$
- Prediction for one input: $\Omega = \{\text{all classes}\}$

An **event** is a subset of $\Omega$ — a set of outcomes we are interested in.

- $A = \{\text{roll is even}\} = \{2, 4, 6\}$
- $B = \{\text{roll} > 3\} = \{4, 5, 6\}$
- $\Omega$ itself (certain event), $\varnothing$ (impossible event)

**Operations** (from section 00): $A \cap B$ = both, $A \cup B$ = either, $A^c$ = not $A$.

For the die: $A \cap B = \{4, 6\}$, $A \cup B = \{2, 4, 5, 6\}$.

## The Axioms (Kolmogorov)

A function $P$ assigning numbers to events is a valid probability if:

1. **Non-negativity:** $P(A) \ge 0$ for every event $A$.
2. **Normalization:** $P(\Omega) = 1$ (something must happen).
3. **Additivity:** for mutually exclusive events ($A \cap B = \varnothing$): $P(A \cup B) = P(A) + P(B)$ — extends to countably many events.

Everything provable about probability follows from these three rules. They are what makes the theory consistent — a probability is a *measure* that allocates total mass 1.

**Immediate consequences:**

$$
P(\varnothing) = 0
\qquad
P(A^c) = 1 - P(A)
\qquad
P(A) \le 1
$$

($P(A) + P(A^c) = P(\Omega) = 1$ gives the second.)

## Equally Likely Outcomes

When every outcome in a finite $\Omega$ is equally likely (each $1/|\Omega|$):

$$
P(A) = \frac{|A|}{|\Omega|}
$$

The die: $P(\text{even}) = 3/6 = 1/2$, $P(>3) = 3/6 = 1/2$, $P(\text{roll} = 6) = 1/6$.

**Worked example — two dice.** $\Omega$ has $6 \times 6 = 36$ equally likely pairs.

- $P(\text{sum} = 7)$: pairs $(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)$ → $6/36 = 1/6$.
- $P(\text{both even})$: each die 3 ways → $9/36 = 1/4$.

The counting technique (identify $\Omega$, count favorable outcomes) is elementary but appears inside AI reasoning about discrete spaces — vocabulary-level probabilities in language models are sums over sets of tokens.

## General Addition Rule

For arbitrary events (not necessarily exclusive):

$$
\boxed{\;P(A \cup B) = P(A) + P(B) - P(A \cap B)\;}
$$

Why: adding $P(A)$ and $P(B)$ counts $A \cap B$ twice, so subtract it once — inclusion–exclusion from section 00, now with probabilities.

**Worked example.** Die: $A$ = even, $B$ = $>3$.

$$
P(A) = \tfrac12,\; P(B) = \tfrac12,\; P(A\cap B) = P(\{4,6\}) = \tfrac13
$$

$$
P(A \cup B) = \tfrac12 + \tfrac12 - \tfrac13 = \tfrac23
$$

Direct count: $|\{2,4,5,6\}|/6 = 4/6 = 2/3$ ✓

**Disjoint case:** if $A \cap B = \varnothing$ the intersection term is $0$ — the axiom of additivity.

## Conditional Probability Preview

$P(A \mid B)$ = probability of $A$ *given that $B$ happened* — defined formally in the next chapter, but the intuition: restrict attention to $B$ and rescale so $B$ has total probability 1.

**Worked example.** $P(\text{even} \mid >3)$: within $\{4,5,6\}$, two are even → $2/3$. Compare the unconditional $1/2$ — knowing $B$ changed the answer. This dependence between information and probability is the entire subject of chapter 02.

## Probability as a Model (Honest Notes)

Probability statements need a *source of uncertainty*:

- **Frequentist:** relative frequency in repeated trials (die rolls, coin flips). $P(A)$ = long-run fraction.
- **Subjective:** degree of belief given evidence — valid for one-time events ("probability this model is better").

AI uses both: a model's predicted probability $P(\text{cat} \mid \text{image})$ is interpreted as a calibrated long-run frequency *and* as a belief to act on. The mathematics does not care which interpretation; using probabilities *as if* they obey the axioms is the commitment.

Known limitations worth remembering:

- Real data are rarely *independent and identically distributed* — probabilities estimated from yesterday's data may not hold for today's.
- A model's output probability is only as good as its training distribution — overconfidence on out-of-distribution inputs is a failure mode precisely because the numbers still "look like" probabilities.

## Connection to AI

- **Every classification output is a probability vector.** "90% cat" means $P(\text{cat} \mid \text{input}) = 0.9$ — an event in the sample space {cat, dog, bird}.
- **Accuracy vs probability.** A model can be right often while its stated probabilities are wrong (miscalibration) — probability measures more than the argmax.
- **Data augmentation and sampling** reason about event frequencies; class weights adjust $P(\text{class})$.
- **The axioms are what code assumes.** Softmax outputs are valid probabilities *because* they are non-negative and sum to 1 — literally axioms 1 and 2.

## Common Mistakes

- $P(A \cup B) = P(A) + P(B)$ **without** subtracting the intersection (only valid for disjoint events).
- Counting outcomes unequally likely as if they were equally likely (e.g., card hands vs individual cards).
- Treating $P(A \cap B)$ and $P(A \mid B)$ as the same — joint versus conditional; the second renormalizes by $P(B)$.
- Assuming $P(A) = 0$ implies $A = \varnothing$ — a probability-zero event can be possible (e.g., continuous outcomes; chapter 03).
- $P(A \mid B) \ne P(B \mid A)$ — the direction of conditioning matters (Bayes' theorem reconciles them).

## Exercises

1. A card is drawn from a standard 52-card deck. Compute $P(\text{ace})$, $P(\text{heart})$, $P(\text{ace or heart})$.
2. Two fair coins are tossed. List $\Omega$ and compute $P(\text{both heads})$, $P(\text{at least one head})$.
3. Compute $P(A \cup B)$ from the axioms' addition rule for $P(A) = 0.4$, $P(B) = 0.5$, $P(A \cap B) = 0.2$.
4. Given a die roll, compute $P(A \cap B)$ for $A$ = even, $B$ = $>3$, then verify the general addition rule against your earlier count of $A \cup B$.
5. Show that $P(A^c) = 1 - P(A)$ follows from axioms 2 and 3.
6. In words: why is $P(\varnothing) = 0$ forced by axiom 1 plus additivity rather than being a separate rule?
