# Conditional Probability and Bayes' Theorem

## Why It Matters for AI

Almost every probability that matters in AI is conditional: $P(\text{class} \mid \text{features})$, $P(\text{next token} \mid \text{previous tokens})$, $P(\theta \mid \text{data})$. Learning *is* conditioning — starting from a prior belief and updating it with evidence. This chapter defines the operation, derives the multiplication rule, and derives Bayes' theorem from nothing but the definition.

## Definition

For events with $P(B) > 0$:

$$
\boxed{\;P(A \mid B) = \frac{P(A \cap B)}{P(B)}\;}
$$

Read: probability of $A$ among outcomes in which $B$ occurred. Knowing $B$ shrinks the sample space from $\Omega$ to $B$, and probabilities rescale so that $P(B \mid B) = 1$.

**Worked example.** Die: $P(\text{even} \mid > 3)$.

$A \cap B = \{4, 6\}$, so $P(A \cap B) = 2/6$; $P(B) = 3/6$:

$$
P(A \mid B) = \frac{2/6}{3/6} = \frac{2}{3}
$$

which matches the direct count within $\{4,5,6\}$ from chapter 01.

## Multiplication Rule

Rearranging the definition ($P(B) > 0$):

$$
\boxed{\;P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)\;}
$$

Joint probability = conditional × marginal of the conditioning event. Both rearrangements express the same product; which one you use depends on which conditional is given.

**Worked example.** Two cards drawn without replacement: $P(\text{both aces})$.

- First card ace: $4/52$.
- Second ace *given* first was ace: $3/51$.
- Product: $\frac{4}{52}\cdot\frac{3}{51} = \frac{12}{2652} = \frac{1}{221} \approx 0.0045$.

The conditioning updates the second draw — information flows through probabilities.

## Independence

Events $A$ and $B$ are **independent** when

$$
P(A \cap B) = P(A)P(B)
$$

Equivalently (when $P(B)>0$): $P(A \mid B) = P(A)$ — knowing $B$ does not change $A$'s probability.

**Worked example — independent.** Two coin flips: $P(\text{both } H) = \frac12 \cdot \frac12 = \frac14 = P(H)P(H)$ ✓ — the second flip does not care about the first.

**Worked example — dependent.** Cards without replacement: $P(\text{ace}_2) = 4/52$ but $P(\text{ace}_2 \mid \text{ace}_1) = 3/51 \ne 4/52$. With replacement they would be independent.

**Warning:** $P(A \cap B) = P(A)P(B)$ can hold for genuinely dependent events by numerical coincidence; independence is an assumption about the mechanism, not something to infer from one matching number. And two events that are **disjoint with positive probabilities** are never independent: $P(A \cap B) = 0 \ne P(A)P(B) > 0$ — knowing one occurred tells you the other did not.

## Total Probability

If $B_1, \ldots, B_n$ **partition** $\Omega$ (mutually exclusive, exhaustive), then for any $A$:

$$
P(A) = \sum_{i=1}^{n} P(A \mid B_i)\,P(B_i)
$$

Every path to $A$ through some $B_i$ is counted once. This is the denominator in Bayes' theorem — the unconditional probability assembled from conditional pieces.

## Bayes' Theorem

Two expressions for the same joint probability:

$$
P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)
$$

Divide both sides by $P(B)$ (positive):

$$
\boxed{\;P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}\;}
$$

With the partition form for $P(B)$:

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{\sum_i P(B \mid A_i)\,P(A_i)}
$$

**Terminology:**

| term | name | reading |
|---|---|---|
| $P(A)$ | **prior** | belief before seeing $B$ |
| $P(A \mid B)$ | **posterior** | belief after seeing $B$ |
| $P(B \mid A)$ | **likelihood** | how well $B$ fits the hypothesis $A$ |
| $P(B)$ | **evidence** | total probability of the observation |

**The derivation in full** (no theorem imported — just algebra):

1. Definition: $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$.
2. Definition with roles swapped: $P(B \mid A) = \dfrac{P(A \cap B)}{P(A)}$, so the joint is $P(A \cap B) = P(B \mid A)\,P(A)$.
3. Substitute (2) into (1): $P(A \mid B) = \dfrac{P(B \mid A)\,P(A)}{P(B)}$. $\square$

**Worked example — a test with base rates.** A disease affects $1\%$ of people ($P(D) = 0.01$). A test: $99\%$ of sick people test positive ($P(+ \mid D) = 0.99$), and $5\%$ of healthy people test positive ($P(+ \mid H) = 0.05$). You test positive — what is $P(D \mid +)$?

Prior: $P(D) = 0.01$, $P(H) = 0.99$.

Evidence: $P(+) = P(+\mid D)P(D) + P(+\mid H)P(H) = 0.99(0.01) + 0.05(0.99) = 0.0099 + 0.0495 = 0.0594$.

$$
P(D \mid +) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} = \frac{1}{6} \approx 0.167
$$

A positive result raises the probability from $1\%$ to about $16.7\%$ — much lower than intuition suggests, because the disease is rare: false positives from the large healthy population rival true positives. **The prior matters.**

## Independence in Bayesian Terms

If $A \perp B$: $P(A \mid B) = P(A)$ — observing $B$ leaves the belief unchanged. Conditioned on the right information, independence often appears (a token's identity may be independent of the document's font); the art of modeling is choosing what to condition on.

## Connection to AI

- **Bayes' rule = learning.** Parameters $\theta$, data $D$: $P(\theta \mid D) = \frac{P(D \mid \theta)P(\theta)}{P(D)}$. MAP estimation (section 04) is maximizing the numerator — prior plus likelihood.
- **Classifiers output posteriors.** $P(\text{class} \mid \text{features})$ is precisely what logistic regression and neural classifiers estimate; the denominator (evidence) is the normalizer of softmax.
- **Naive Bayes** classifies by assuming features independent given the class — multiplying conditionals — a direct application of this chapter.
- **Calibration** asks: among cases where the model said $P = 0.8$, does frequency confirm $0.8$? A Bayes-consistency question.
- **Base-rate neglect** (the test example) is a known human failure; models trained with imbalanced data re-learn the same arithmetic — accuracy can look high while the posterior for the rare class stays tiny.

## Common Mistakes

- Swapping the direction: $P(A \mid B) \ne P(B \mid A)$ — Bayes exists precisely to translate between them.
- Forgetting to divide by $P(B)$ (or the evidence term): posteriors must sum to 1 over the partition.
- Using the multiplication rule without conditioning when events are dependent (cards without replacement).
- Calling events independent because their intersection probability *equals* the product by numerical accident — check the mechanism, or the conditional.
- Ignoring the prior when it is small (the test example): likelihood alone misleads.

## Exercises

1. Compute $P(A \mid B)$ for $P(A \cap B) = 0.12$, $P(B) = 0.3$.
2. Two dice: compute $P(\text{sum} = 7 \mid \text{first die is even})$ by counting within the restricted sample space.
3. Verify the multiplication rule on problem 2: compute $P(\text{sum}=7 \cap \text{first even})$ and $P(\text{first even})$, then their conditional.
4. A bag holds 3 red and 2 blue balls; draw two without replacement. Compute $P(\text{both red})$ using conditional probabilities.
5. Use Bayes' theorem: $P(B \mid A) = 0.8$, $P(A) = 0.25$, $P(B) = 0.4$. Find $P(A \mid B)$.
6. Redo the medical-test example with prevalence $P(D) = 0.001$ (0.1%). What is $P(D \mid +)$ now, and what does the change illustrate?
7. Give an example of two events that are mutually exclusive with positive probabilities, and explain why they cannot be independent.
