# The Neuron and Activation Functions

## Why It Matters for AI

Everything a deep network does is built from one computational atom: a weighted sum plus a bias, squashed by a nonlinear function. The weighted sum you already know (linear/logistic regression — section 07); the squashing function is what this chapter chooses carefully. The choice shapes gradients, training speed, and what the network can represent.

## The Neuron

For input $\mathbf{x} \in \mathbb{R}^d$, weights $\mathbf{w} \in \mathbb{R}^d$, bias $b \in \mathbb{R}$:

$$
z = \mathbf{w}^\top \mathbf{x} + b \qquad\text{(pre-activation)}
$$
$$
a = f(z) \qquad\text{(activation output)}
$$

- $z$: the linear score — *identical* to logistic regression's logit (section 07).
- $f$: the **activation function** — nonlinear by design (see below).
- $a$: what passes to the next neuron.

A single neuron with $f = \sigma$ and BCE loss *is* logistic regression. Networks differ by **depth** and **width**, not by a new atomic operation.

## Why Nonlinearity Is Not Optional

**Claim:** composing affine maps without activation gives an affine map.

$$
\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1,\quad
\mathbf{z}_2 = W_2 \mathbf{z}_1 + \mathbf{b}_2
\;\Longrightarrow\;
\mathbf{z}_2 = \underbrace{W_2 W_1}_{\text{one matrix}} \mathbf{x} + \underbrace{(W_2 \mathbf{b}_1 + \mathbf{b}_2)}_{\text{one bias}}
$$

Ten stacked linear layers = one linear layer (with different parameters). No nonlinearity ⇒ no depth benefit ⇒ only hyperplane boundaries (section 07's decision boundary, forever). The activation is precisely what makes $\mathbf{z}_2$ *not* reducible, letting deep networks bend, fold, and carve curved regions.

## The Standard Activations

**Sigmoid** $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ — from section 07: range $(0,1)$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$, max slope $1/4$ at $z = 0$, saturating for $|z|$ large. Historically central (interpretable probability); now rare in *hidden* layers — saturation squeezes gradients (below).

**Tanh** $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ — range $(-1, 1)$, zero-centered (output mean nearer 0 — usually friendlier than sigmoid's positive mean), $\tanh'(z) = 1 - \tanh^2(z)$, max slope $1$ at $0$, also saturating.

**ReLU** $\mathrm{ReLU}(z) = \max(0, z)$

$$
\mathrm{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}
$$

(at $z = 0$ the derivative is undefined; frameworks assign a subgradient — section 02's noted theory gap, harmless in practice). Properties:

- **Unsaturated for $z > 0$:** gradient $= 1$ exactly — no shrinkage through depth (the main reason ReLU dominates hidden layers).
- **Sparse:** exactly zero output/gradients for negative inputs — computation and representation both economize.
- **Cheap:** a comparison, not exponentials.
- **Dying ReLU:** if a neuron's input range goes negative (bad init/large rate), it outputs $0$ forever with $0$ gradient — dead, no recovery. Leaky variants ($\max(0.01z, z)$) exist for this.

**Output-layer activations** are chosen by the *task's likelihood* (section 06/07), not by habit:

| task | loss (section) | output activation |
|---|---|---|
| binary classification | BCE | sigmoid |
| multiclass | cross-entropy | softmax (section 09) |
| regression | MSE / Gaussian NLL | identity (none) |

## The Vanishing Gradient (Honest Version)

Backprop multiplies activation derivatives along the path (chapter 03). Sigmoid's slope is at most $0.25$; through $L$ sigmoid layers a gradient carries a factor up to $(0.25)^L$:

$$
L = 6 \Rightarrow 0.25^6 \approx 2.4 \times 10^{-4};\qquad L = 12 \Rightarrow \approx 6 \times 10^{-8}
$$

Early layers' gradients become numerically negligible — the network's bottom stops learning while the top still moves (**vanishing gradients**). ReLU's derivative of $1$ avoids the multiplicative squeeze on active paths (its own failure mode is the dead neuron). This is the practical reason architectures shifted from sigmoid to ReLU in the 2010s — a one-line activation change with field-wide consequences (modern variants: GELU, SiLU — smooth approximations of ReLU, same rationale).

## Connection to AI

- **Section 07 was this chapter with one layer:** logistic regression = one neuron, sigmoid, BCE. Everything you derived there ($\partial \ell / \partial z = \sigma(z) - y$) is reused at every output neuron.
- **Activation choice is architecture:** attention uses softmax internally (section 09), GPT-family blocks use GELU in feed-forward parts — same slot, different $f$.
- **Probabilistic outputs:** sigmoid/softmax activations make $a$ interpretable as probability *because* the loss is the matching likelihood — activations and losses are chosen as a pair.
- **Depth needs healthy gradients:** residual connections (section 09) exist largely to keep gradient paths near-1 across many layers — vanishing gradients' architectural cure.

## Common Mistakes

- Stack linear layers and expect expressiveness — the collapse above; nonlinearity is the entire point of depth.
- Softmax on a single output (redundant: $\sigma$ of one logit already) or sigmoid per class for mutually exclusive labels (wrong normalization — use softmax).
- Choosing ReLU at the output for regression — it cannot produce negative values; use identity.
- Ignoring dead neurons: all activations stuck at $0$ after init — check activation statistics during training.
- Assuming sigmoid outputs are calibrated probabilities without BCE training (section 07's calibration point).
- Forgetting that tanh's range $(-1,1)$ implies different initialization scales than ReLU.

## Exercises

1. Show algebraically that $\mathbf{z}_2 = W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$ collapses to a single affine map — name the two combined parameters.
2. Compute $f$ and $f'$ at $z = 0$ for: sigmoid, tanh, ReLU (state the subgradient convention at $0$ for ReLU).
3. Why can't a network of 20 sigmoid-only layers propagate gradients well? Quantify the worst-case factor.
4. A regression output head predicts house prices in $[2\times 10^5, 10^6]$. Which output activation and loss? Justify in one sentence.
5. Compute $\tanh(1)$ and $\tanh'(1)$ ($\tanh(1) \approx 0.7616$).
6. State the dying-ReLU failure and one mitigation.
