# Objective and Loss Functions

## Why It Matters for AI

"Training" has no content until you say *what is being minimized*. The **objective function** is the quantity that defines success for the optimizer; the **loss** measures one example's error, and the objective aggregates losses over data. Choosing the objective is choosing the problem — every model family is distinguished less by its architecture than by what it optimizes.

## Objective Functions

An **objective** $J: \mathbb{R}^d \to \mathbb{R}$ maps parameters to one number:

$$
J(\theta) \to \text{scalar}, \qquad \theta \in \mathbb{R}^d
$$

- **Minimization** (usual): find $\theta^*$ with $J(\theta^*) \le J(\theta)$ for all $\theta$.
- **Maximization**: equivalent by negation: $\max J = \min(-J)$. (Reward maximization in RL is written as maximizing — same machinery, flipped sign.)

The solution $\theta^*$ is a **minimum** of $J$. Goals: **global minimum** ($\theta^*$ beats *every* point) versus **local minimum** (beats its neighborhood). Section 03 (convexity) explains when the two coincide; the general case is harder than finding *a* local minimum — the section's later chapters return to this.

## Loss vs Objective

Two words, two levels:

| term | scope | example |
|---|---|---|
| **loss** $L(\theta; x, y)$ | one example (or one term) | squared error $(\hat y - y)^2$ for a single point |
| **objective** $J(\theta)$ | the whole problem | average loss over the training set |

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^{n} L(\theta; x_i, y_i)
$$

"Loss function" is sometimes used loosely for both; the distinction matters because SGD (chapter 04) replaces the sum over $n$ terms with a sample of terms — possible only because the objective is an *average of losses*.

## Where Losses Come From

Section 04 established the principle: **the principled loss is the negative log-likelihood of the assumed model**.

| assumption about noise/labels | loss |
|---|---|
| Gaussian output noise | $L = (y - f(x))^2$ (squared error) |
| Bernoulli/binary label | $L = -[y\ln\hat p + (1-y)\ln(1-\hat p)]$ (binary cross-entropy) |
| categorical label / next token | $L = -\ln \hat p_{\text{true}}$ (cross-entropy) |

Architectures change $f$; the likelihood choice fixes the loss; optimization minimizes the result.

**Worked example — the pieces of a regression objective.** Linear model $\hat y = w x + b$, squared loss, 3 points from section 01:

$$
J(w, b) = \frac{1}{3}\sum_{i=1}^{3}\left(y_i - (wx_i + b)\right)^2
$$

$\theta = (w, b)$, two parameters; the surface $J(w,b)$ is a bowl-shaped quadratic (chapter 03 proves it is convex) whose bottom is exactly the normal-equations solution from section 01. Same minimum, two derivations.

## Desiderata of a Good Objective

- **Differentiable** (or nearly): gradient methods apply (section 02). ReLU networks are piecewise smooth — fine in practice.
- **Aligned with the real goal**: if you care about ranking, MSE may be the wrong objective; the loss *is* the spec.
- **Averagable over data**: additive structure $J = \sum_i L_i$ enables minibatching (chapter 04).
- **Well-conditioned**: smooth scaling across parameter directions speeds convergence (standardization from section 04 helps here).

## Connection to AI

- **Different objectives, different models.** Linear regression minimizes squared error; logistic regression minimizes cross-entropy; PCA (as an optimization problem) minimizes reconstruction error; language models minimize token cross-entropy (section 10). One operation — minimization — across the field.
- **Multi-objective reality.** "Loss" in a real system may combine terms: $\mathcal{L} = L_{\text{task}} + \lambda_1 L_{\text{regularization}} + \lambda_2 L_{\text{auxiliary}}$ — all weighted scalars packed into one $J$.
- **Reward as negative loss** in reinforcement learning: maximize $R$ = minimize $-R$; the objective stays scalar.
- **Evaluation metrics ≠ training objective.** Accuracy is not differentiable, so training uses cross-entropy as a smooth proxy — an acknowledged mismatch, not an accident.

## Common Mistakes

- Treating the training objective as the true goal — improving $J$ while the real metric stalls means the proxy is wrong (or the objective is miscomputed).
- Forgetting the $1/n$ (or not — but be consistent): gradients differ by a constant factor, absorbed into $\eta$; comparisons across codebases often trip on this.
- Writing $J(\theta)$ when the loss depends on data as well — $J$ implicitly averages over the fixed training set; remember the dependence.
- Assuming differentiability everywhere: subgradients handle $L_1$ and ReLU kinks (chapter 02 noted the theory gap; practice assigns a valid slope).
- Confusing minimizing training loss with achieving generalization — chapter 04's estimation gap; regularization (chapter 05) exists for it.

## Exercises

1. Write the objective $J(\theta)$ for logistic regression on $n$ examples with binary cross-entropy per example (definitions from chapter 03 of section 04).
2. For $L(\theta; x,y) = (\hat y - y)^2$ with $\hat y = wx$, express $\partial L/\partial w$ for a single example with $x = 2$, $y = 5$, $w = 1$.
3. Why can SGD average losses over a minibatch? (One sentence, referencing the structure of $J$.)
4. Convert "maximize accuracy on the validation set" into an optimization problem — why is that conversion problematic for gradient methods?
5. Give the loss for a classification model whose softmax assigns $0.01$ probability to the true class.
6. State the difference between a local and a global minimum in one sentence each.
