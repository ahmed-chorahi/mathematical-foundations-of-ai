# Mutual Information

## Why It Matters for AI

Is this feature worth having? Does the prefix tell the model anything about the token? Does the hidden state retain the input's relevant content? **Mutual information** $I(X; Y)$ answers all three as one number: how much knowing $X$ reduces uncertainty about $Y$ — equivalently, how much the two variables *share*. It is built entirely from this section's previous chapters, and it closes the loop: conditioning reduces entropy by exactly the mutual information.

## Definition (Three Equivalent Forms)

$$
\boxed{\;I(X; Y) = H(Y) - H(Y \mid X) = H(X) - H(X \mid Y) = D_{\mathrm{KL}}\big(p(x,y) \,\|\, p(x)\,p(y)\big)\;}
$$

- **First form:** the uncertainty $X$ removes from $Y$ (how much the label becomes predictable from the input).
- **Second form:** symmetric removal — hence the name: the information *mutually* shared (subtracting in either direction gives the same number; see below).
- **Third form:** KL between the true joint and the product-of-marginals joint — "how far is reality from independence?" (chapter 03's machinery).

*Why forms 1 and 2 agree:* both equal $H(X, Y) - H(X) - H(Y)$ by the chain rule of chapter 04 — subtract $H(X \mid Y) = H(X,Y) - H(Y)$ from $H(X)$, or subtract $H(Y \mid X) = H(X,Y) - H(X)$ from $H(Y)$; the result is $H(X,Y) - H(X) - H(Y)$ either way.

*Why form 3 agrees:* expand the KL sum:

$$
\sum_{x,y} p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}
= -\sum_{x,y} p(x,y)\ln p(x,y) + \sum_{x,y} p(x,y)\big[\ln p(x) + \ln p(y)\big]
$$

first term is $H(X,Y)$; the others collapse to $-H(X) - H(Y)$ (marginalizing the joint reproduces each marginal). Total: $H(X,Y) - H(X) - H(Y)$ ✓.

## Worked Example (Chapter 04's Table)

The same joint distribution ($X$ = previous word, $Y$ = next word):

$$
H(Y) = \ln 2 \approx 0.6931, \qquad H(Y \mid X) \approx 0.5004 \text{ nats}
$$

$$
I(X; Y) = 0.6931 - 0.5004 = 0.1927 \text{ nats}
$$

**Verify via form 2:** the table is symmetric, so $H(X) = \ln 2$ and $H(X \mid Y) = H(Y \mid X) = 0.5004$ — same $0.1927$ ✓.

**Verify via form 3:** compare joint with independence product $p(x)p(y) = [0.25, 0.25; 0.25, 0.25]$:

$$
I = 0.4\ln\frac{0.4}{0.25} + 0.1\ln\frac{0.1}{0.25} + 0.1\ln\frac{0.1}{0.25} + 0.4\ln\frac{0.4}{0.25}
$$

$$
= 2\big[0.4\ln 1.6 + 0.1\ln 0.4\big] = 2\big[0.4(0.4700) + 0.1(-0.9163)\big] = 2(0.1880 - 0.0916) = 0.1927 \text{ nats} \;✓
$$

(all three forms agree — verified numerically.)

## Properties

1. **Non-negative:** $I(X; Y) \ge 0$, since it is a KL divergence (chapter 03) — equivalently $H(Y \mid X) \le H(Y)$, which finally *proves* chapter 04's "conditioning reduces entropy". Equality iff $X \perp Y$ (KL is zero iff joint = product of marginals).
2. **Symmetric:** $I(X; Y) = I(Y; X)$ — unlike KL, the argument order doesn't matter (it measures a relation, not a direction). (KL's asymmetry hides inside form 3 only through the symmetric joint-vs-product comparison.)
3. **Bounded:**
   $$
   0 \;\le\; I(X; Y) \;\le\; \min\big(H(X),\, H(Y)\big)
   $$
   — you cannot learn more than you didn't know. Max = $\min$ iff one variable determines the other (deterministic relationship on the shared support).
4. **Zero exactly at independence** — the defining test.
5. **Chain rule for MI:** $I(X; Y, Z) = I(X; Z) + I(X; Y \mid Z)$ (conditional mutual information) — used in feature analysis, not developed here.

**Worked bound check:** $I = 0.1927 \le \min(0.6931, 0.6931)$ ✓.

## Reading the Extremes

| situation | $I(X; Y)$ | meaning |
|---|---|---|
| independence | $0$ | $X$ irrelevant to $Y$ — drop the feature |
| $Y = f(X)$ deterministic | $H(Y)$ (max) | $X$ fully determines $Y$ — $H(Y \mid X) = 0$ |
| partially informative | between | each nat of $I$ is one nat less surprise in predicting $Y$ given $X$ |

## Connection to AI

- **Feature selection:** rank features by $I(X_j; Y)$; irrelevant features have near-zero MI with the label. Estimating MI from samples is hard (an active research area) — but the *definition* is the target every estimator chases.
- **"Is attention/hidden state keeping what matters?"** — probing questions ("does layer 7 encode syntax?") are informal statements about $I(\text{hidden state}; \text{property})$; the **information bottleneck** framework formalizes training as maximizing $I(\text{representation}; Y)$ while minimizing $I(\text{representation}; X)$ — compress input, retain label-relevant content.
- **Language models:** $I(W_t; W_{<t}) = H(W_t) - H(W_t \mid W_{<t})$ — exactly how much the prefix reduces next-token uncertainty; perplexity improvements *are* MI discoveries about the corpus (section 10).
- **Redundancy in data:** $I$ between two views/augmentations of the same example motivates contrastive learning (pull together what shares information).
- **Naive Bayes' assumption:** features conditionally independent given the label — a *modeling* statement about which MI terms may be dropped from $H(X_1, \ldots, X_d \mid Y)$.
- **Independence tests** (statistical: does $X$ tell us anything about $Y$?) are MI questions — statistics meets information theory.

## Common Mistakes

- Confusing $I(X; Y)$ with $H(X \mid Y)$ or $H(Y \mid X)$ — shared vs remaining are complementary: $I + H(Y \mid X) = H(Y)$.
- Assuming large $H(Y \mid X)$ means large $I$ — opposite: *small* $H(Y\mid X)$ means *large* $I$ (given $H(Y)$ fixed).
- Claiming $I$ measures causal influence — it measures statistical dependence only ($X$ and $Y$ can share information through a common cause).
- Comparing MI across different alphabets/scales without care — the ceiling is $\min(H(X), H(Y))$; a near-ceiling number means near-determinism, not "better model".
- Computing $I$ from a *model's* predicted joint instead of data — then you measure the model's beliefs, not the world's dependence.

## Exercises

1. $X \perp Y$: what is $I(X; Y)$ by each of the three definitions? (One line each.)
2. $Y$ deterministic given $X$, $H(Y) = 2$ nats: find $I(X; Y)$ and $H(Y \mid X)$; check the upper bound.
3. Chapter's table: compute $I(X; Y)$ as $H(X) - H(X \mid Y)$ using $H(X) = \ln 2$ and the chain rule value $H(X \mid Y) = H(X,Y) - H(Y)$.
4. Compute $I(X; Y)$ for the uniform joint $p(x,y) = \tfrac14$ over 4 outcomes, and interpret.
5. $H(Y) = 1.0$ nats, $H(Y \mid X) = 0.7$ nats: state $I(X;Y)$, and say in words how much a predictor using $X$ can beat one ignoring $X$ (in expected log-loss).
6. Feature A has $I(A; Y) = 0$; feature B has $I(B; Y) = 0.3$ nats with $H(Y) = 0.9$ nats. Which is useless and why? What is $H(Y \mid B)$?
