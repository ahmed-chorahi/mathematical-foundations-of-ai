# Sampling and Temperature

## Why It Matters for AI

A trained model gives you a distribution — the last chapter's pipeline ends at $P(w_{t+1} \mid w_{\le t})$. Turning that distribution into *text* requires a generation policy: always take the best token (deterministic), or draw from the distribution (random)? Sharpen it, flatten it? **Temperature** is the single parameter that reshapes the distribution before drawing; **top-$p$ (nucleus) sampling** truncates its tail. Both are one formula and one sort away — and both are explained completely by section 06's entropy.

## The Generation Loop

For each step: run chapter 01's pipeline $\to$ logits $\boldsymbol{\ell}_t$ $\to$ (optional reshape, below) $\to$ select a token $\to$ append to prefix $\to$ repeat. Selection policies:

- **Greedy:** $\arg\max_j \ell_{t,j}$ — deterministic, often repetitive over long runs (locally optimal, globally dull — a mode-collapsed trajectory, section 06's mode language).
- **Sampling:** draw $w_{t+1} \sim P(\cdot)$ — diverse, but inherits every quirk of the distribution's tail.

## Temperature: Resharping the Distribution

Given logits $\boldsymbol{\ell}$ and temperature $T > 0$:

$$
\boxed{\;P_T(j) = \frac{e^{\ell_j / T}}{\sum_{j'} e^{\ell_{j'} / T}}\;}
$$

— softmax of the *scaled* logits (chapter 02 of section 09 divided by $\sqrt{d_k}$ for variance control; here $T$ deliberately *sets* sharpness for generation).

**Worked example.** Logits $(2.0, 1.0, 0.1)$ (chapter 01's), vocabulary $(\texttt{cat}, \texttt{dog}, \texttt{the})$:

| $T$ | scaled logits | $P_T$ |
|---|---|---|
| $0.5$ | $(4.0, 2.0, 0.2)$ | $(0.864,\ 0.117,\ 0.019)$ |
| $1$ (identity) | $(2.0, 1.0, 0.1)$ | $(0.659,\ 0.242,\ 0.099)$ |
| $2$ | $(1.0, 0.5, 0.05)$ | $(0.502,\ 0.304,\ 0.194)$ |

(verified numerically; each row sums to 1.) Every temperature **preserves the ranking** — $e^{x}$ is monotone, so $\ell_i > \ell_j \Rightarrow P_T(i) > P_T(j)$ always; only *gaps* change.

**Limits (exact):**

$$
T \to 0^+:\ P_T \to \text{one-hot on } \arg\max \ (\text{greedy});\qquad
T \to \infty:\ P_T \to \text{uniform } (1/V \text{ each})
$$

**Entropy reading (section 06, chapter 01).** Lower $T$ concentrates mass: $H(P_T)$ falls toward $0$ (certain, predictable, repetitive output). Higher $T$ flattens: $H(P_T)$ rises toward $\ln V$ (maximum uncertainty — random-ish output). **Temperature is an entropy dial:** you choose where on the $[0, \ln V]$ scale generation should sit.

**Why it works mechanically:** dividing logits by $T<1$ *magnifies* differences after exponentiation ($e^{\ell/T}$ amplifies gaps); $T>1$ shrinks them. Same softmax, re-parameterized spread — the $1/\sqrt{d_k}$ lesson reversed: there we removed *accidental* spread; here we add *intentional* spread.

**Defaults and practice:** $T = 1$ = trust the model as trained (the distribution PPL evaluated); factual/analytical tasks often $T \le 0.3$; creative tasks $T \ge 0.7$. Temperature does not change the model — only the reading of it.

## Top-$p$ (Nucleus) Sampling: Truncating the Tail

Temperature reshapes everything; often you want to *drop* the long tail of implausible tokens entirely. **Nucleus sampling:** sort $P(j)$ descending, keep the **smallest set** whose cumulative probability reaches $p$, renormalize, sample.

**Worked example.** Probabilities $(0.50, 0.30, 0.15, 0.05)$, $p = 0.9$: cumulative $0.50, 0.80, 0.95 \ge 0.9$ — keep the first three, renormalize by $0.95$:

$$
(0.526,\ 0.316,\ 0.158)
$$

The $0.05$ token is excluded — the sample cannot draw garbage the model itself ranks as tail noise. $p = 1.0$ disables truncation; $p \to 0$ approaches greedy.

- **Adaptive set size:** when the model is confident, the nucleus is small (one or two tokens cover $0.9$); when uncertain, it's large — the cutoff breathes with the distribution, unlike fixed top-$k$.
- **Order of operations:** temperature is applied **before** the nucleus cut (reshape, then truncate, then renormalize); the reverse changes the selected set — a real implementation detail.

**Combined with greedy:** common stacks are "greedy for the last-token/short answers", "$T \approx 0.7$ + top-$p = 0.9$" for chat. These are policy choices on top of the same $P$.

## Honest Limits

- Sampling cannot fix a *wrong* distribution — temperature trades fidelity for variety; if the model's $P$ misjudges truth, every $T$ inherits it (sharper $T$ just commits to the error faster).
- **Repetition loops** arise partly from the model's own distribution being peaked on self-continuations — temperature raises diversity but does not alter context; repetition penalties modify logits (another policy, same pipeline).
- **Evaluation stays at $T = 1$:** PPL (chapter 03) measures the model, not your sampling settings; a "better-sounding" sample at $T = 0.5$ has not improved the likelihood.
- **Stochasticity is a feature, not noise:** for open-ended generation, multiple valid outputs exist (real conditional entropy $> 0$ — section 04); sampling honestly represents that ambiguity; greedy hides it.

## Connection to AI

- **Chat behavior is largely these knobs:** deterministic "please summarize" runs vs creative runs differ in exactly $T$ and top-$p$ — the user-facing edge of section 06's entropy.
- **Temperature shows up in training too:** RL fine-tuning stages sample trajectories from the policy at chosen temperatures — same formula, different context (beyond this course's scope; the symbol recurs).
- **The entropy dial is principled:** with $H(P_T)$ computable (section 06's formula), you can *measure* how diverse your sampling is — settings become quantitative, not vibes.
- **Consistency requirements:** beam search / constrained decoding (forcing syntax) are alternative selection policies on the same logits — again: pipeline fixed, policy chosen.
- **Section 09's $1/\sqrt{d_k}$ and this $1/T$** are the two places softmax input scale is deliberate: one for *trainability*, one for *behavior* — knowing which is which is the chapter's core lesson.

## Common Mistakes

- Temperature $0$ exactly (division by zero) — implementations use greedy when $T \approx 0$; the limit exists, the formula doesn't at $T = 0$.
- Interpreting $T = 0.7$ as "70% confident" — $T$ scales logit *gaps*, not confidence directly; probabilities move per-entry.
- Applying temperature after top-$p$ (or double-applying per layer/step) — order matters; apply once to logits before truncation.
- Reporting sampled-output quality as model quality — evaluation uses $T = 1$ likelihoods (chapter 03's warning).
- Expecting temperature to add *new* knowledge — ranking is preserved (chapter's key fact): $T$ redistributes probability among what the model already ranks.
- Comparing runs at different $T$ as if only the model changed — the policy changed too.

## Exercises

1. Logits $(1, 0)$: compute $P_T$ at $T = 0.5$ and $T = 2$ (3 decimals); confirm the ranking is unchanged.
2. Limits: state $\lim_{T \to 0}$ and $\lim_{T \to \infty}$ of $P_T$; give both entropies (section 06's units).
3. Top-$p$: probabilities $(0.4, 0.3, 0.2, 0.1)$, $p = 0.8$ — which tokens are kept, and renormalized probabilities?
4. Temperature-entropy connection: for fixed logits, does raising $T$ raise or lower $H(P_T)$? One-sentence justification using the $T \to$ limits.
5. Chapter 01's logits with $T = 0.5$: compute the probability of the top token and the expected "surprise" (self-information, section 06) if it is correct: $-\ln 0.864$.
6. Why can't any temperature improve PPL? (One sentence: what PPL measures vs what $T$ changes.)
