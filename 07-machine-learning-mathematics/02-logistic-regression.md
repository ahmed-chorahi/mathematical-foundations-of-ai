# Logistic Regression

## Why It Matters for AI

Classification — spam or not, cat or dog, token or not — needs probabilities in $[0, 1]$, which a linear function cannot produce. **Logistic regression** keeps the linear predictor $\eta = w^\top x + b$ and passes it through the **sigmoid** to get a probability, then trains with cross-entropy (chapter 02 of this section's information-theory neighbor, section 06). It is the canonical *generalized linear model*, the ancestor of every neural classifier, and the place where "output a distribution, train with negative log-likelihood" becomes explicit.

## The Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Properties (all verified by calculus in section 02):

1. **Range** $(0, 1)$ — a probability for any real input; $\sigma(0) = 1/2$, $\sigma(z) \to 1$ as $z \to +\infty$, $\to 0$ as $z \to -\infty$.
2. **Monotone increasing** — bigger evidence $\Rightarrow$ bigger probability.
3. **Symmetry:** $\sigma(-z) = 1 - \sigma(z)$ (check: $\sigma(-z) = \frac{1}{1+e^{z}} = \frac{e^{-z}}{e^{-z}+1} = 1 - \sigma(z)$).
4. **Derivative:** $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$ — the clean identity from section 02 (sigmoid chapter), reused constantly in backpropagation (section 08).
5. **Log-odds linearity:** $\ln\frac{\sigma(z)}{1 - \sigma(z)} = z$ — the logit link: the model says the *log-odds* are linear in the features, not the probability itself.

## The Model

For binary label $y \in \{0, 1\}$:

$$
\boxed{\;q_\theta(x) = P(y = 1 \mid x) = \sigma(w^\top x + b)\;}
$$

**Decision boundary:** predict 1 iff $q \ge \tfrac12$ iff $w^\top x + b \ge 0$ — a hyperplane (the linear structure survives; the probability wraps around it). $\|w\|$ controls the boundary's confidence gradient (steepness in probability space).

**Worked example.** $w = (2, -1)$, $b = 0$; point $x = (1, 1)$: $z = 2 - 1 = 1$, $q = \sigma(1) = 0.7311$. Point $(0, 0)$: $z = 0$, $q = 0.5$ (on the boundary). Point $(-1, 2)$: $z = -4$, $q = \sigma(-4) = 0.0180$ — confidently class 0 (check via symmetry: $1 - \sigma(4) = 1 - 0.9820 = 0.0180$ ✓).

## The Loss: Cross-Entropy in Disguise

Per example, section 06's binary cross-entropy with $q = \sigma(z)$:

$$
\ell(w, b; x, y) = -\big[y \ln q + (1-y)\ln(1-q)\big]
$$

Substitute the sigmoid's log identities ($\ln \sigma(z) = -\ln(1 + e^{-z})$, $\ln(1 - \sigma(z)) = -\ln(1 + e^{z})$):

$$
\ell = \ln(1 + e^{-z}) \ \ (y = 1),
\qquad
\ell = \ln(1 + e^{z}) \ \ (y = 0)
$$

— the **softplus** of section 05 (convexity chapter: $\ell''(z) = \sigma(-z)(1 - \sigma(-z)) > 0$), composed with the affine $z = w^\top x + b$: the objective is convex in $(w, b)$ ⇒ gradient descent finds the **unique global optimum** (the linear-regression guarantee, without a closed form).

**The gradient (the fact that makes this model teachable).** Chain rule through the sigmoid:

$$
\frac{\partial \ell}{\partial z} = \sigma(z) - y
\qquad\Longrightarrow\qquad
\boxed{\;\nabla_w \ell = (\sigma(z) - y)\,x,\qquad \frac{\partial \ell}{\partial b} = \sigma(z) - y\;}
$$

*Check for $y = 1$:* $\frac{d}{dz}\ln(1 + e^{-z}) = \frac{-e^{-z}}{1 + e^{-z}} = \frac{-1}{e^{z}+1} = -\sigma(-z) = \sigma(z) - 1$ ✓ (using $\sigma(z) + \sigma(-z) = 1$); *for $y = 0$:* $\frac{d}{dz}\ln(1 + e^{z}) = \sigma(z)$ ✓.

The gradient is **prediction minus truth, times the input** — the universal pattern: error signal $(\sigma(z) - y)$ scales each feature. Every neural-network layer repeats this shape (section 08).

**Worked step.** One example $x = (1, 1)$, $y = 1$, from $\theta = 0$ ($b = 0$): $z = 0$, $\sigma(z) = 0.5$, error $= 0.5 - 1 = -0.5$; with $\eta = 0.1$:

$$
w \leftarrow 0 - 0.1(-0.5)(1, 1) = (0.05, 0.05),\qquad
b \leftarrow 0 - 0.1(-0.5) = 0.05
$$

$z$ becomes $0.1$, $q = \sigma(0.1) = 0.5250$ — probability moved toward the true label $1$; the loss fell (from $\ln 2 \approx 0.693$ to $\ln(1 + e^{-0.1}) \approx 0.644$).

Training on the dataset = averaging these gradients over examples (section 05's $J = \frac1n\sum L_i$; SGD from section 05 chapter 04 for large $n$). **No closed form** exists — the nonlinearity destroys the normal-equations path; this is the standard case, and why optimization is a separate section.

## Separable Data: A Honest Caveat

If a hyperplane perfectly separates the classes, the likelihood keeps improving as $\|w\| \to \infty$ (pushing $z$'s to $\pm\infty$, losses to $0$): **the MLE does not exist** (the supremum is unattained). Symptoms: coefficients blow up, probabilities snap to $0/1$. Remedies — regularization (section 05: ridge makes the objective coercive and the optimum finite) — is exactly why applied logistic regression always regularizes. (The $\|w\| \to \infty$ direction also previews SVM's max-margin logic from section 05's constraints chapter.)

## Connection to AI

- **The first neural network:** a neuron $z = w^\top x + b$ with sigmoid output *is* logistic regression (section 08 stacks them); its loss carries forward unchanged.
- **Multiclass via softmax** (section 09): $\sigma$'s one-dimensional squeeze generalizes to a distribution over $k$ classes; cross-entropy loss identical in form.
- **Probabilistic outputs matter:** unlike a hard classifier, $q$ supports thresholds (move the 0.5 boundary for asymmetric costs), calibration checks, and downstream decision-making — a model's stated confidence is only meaningful if cross-entropy trained it to be truthful (section 06's "strictly proper" point).
- **Gradient pattern recurs:** $(\text{output} - \text{target}) \times \text{input}$ appears at the output layer of every network — memorize the shape here, recognize it everywhere.
- **Generalized linear models:** replacing $\sigma$ with other links (probit, Poisson) gives other regression families — logistic is the template, not an accident.

## Common Mistakes

- Using squared error for classification — it is not the likelihood, is poorly behaved near $q \in \{0,1\}$, and loses the gradient's clean form.
- Feeding probabilities where logits are expected (or vice versa): $\ell$ can be computed stably from $z$ directly ($\ln(1+e^{-z})$) — code that takes logits vs probabilities must be used correctly (numerically, $-\log\sigma(z)$ is evaluated as softplus to avoid overflow).
- Reading $\sigma(z)$ as a calibrated probability *before* training — it is a squash function; calibration comes from fitting with cross-entropy on representative data.
- Expecting the normal equations — no closed form; use GD/Newton-type methods (Newton's method on this convex problem converges fast — beyond scope).
- Ignoring separability blow-up: diverging coefficients are a *model* property (unregularized MLE), not a bug in your code.
- Dropping the intercept: without $b$, the boundary must pass through the origin.

## Exercises

1. Compute $\sigma(0)$, $\sigma(2)$, $\sigma(-2)$; verify $\sigma(2) + \sigma(-2) = 1$ (symmetry). ($\sigma(2) \approx 0.8808$.)
2. For $w = (1, 1)$, $b = -1.5$: which side is $x = (1, 0)$ on? Give $z$ and the predicted probability.
3. Derive $\frac{d}{dz}\ln(1 + e^{-z}) = \sigma(z) - 1$ from the derivative rules (start: quotient rule or $u = e^{-z}$).
4. One GD step: $y = 0$, $x = (2, 3)$, $\theta = 0$, $\eta = 0.1$. Give new $w$ and $b$.
5. Compute the loss for $y = 1$ when $z = -3$ (use $\ln(1 + e^{3}) \approx 3.0486$). What does the large value teach about confident errors?
6. Why is the logistic objective convex while deep networks are not? (One sentence referencing composition rules.)
