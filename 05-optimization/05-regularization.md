# Regularization

## Why It Matters for AI

A model that memorizes the training set is useless — the goal is generalization (section 04's estimation story: total error = bias² + variance + noise). **Regularization** deliberately constrains or perturbs training to reduce variance, typically by adding a penalty to the objective. This chapter derives the L2 and L1 penalties from the optimization side, shows exactly how they modify gradients (L2 turns into a simple multiplicative decay of the weights), gives the ridge closed form, and connects back to MAP priors from section 04.

## The Regularized Objective

$$
J_{\text{reg}}(\theta) = \underbrace{\frac{1}{n}\sum_{i=1}^{n} L(\theta; x_i, y_i)}_{\text{fit the data}} \;+\; \lambda \underbrace{R(\theta)}_{\text{penalize complexity}}
$$

- $\lambda \ge 0$ — the **regularization strength** (hyperparameter, selected on held-out validation data — statistics' estimation hygiene applied to models).
- $R$ — a complexity measure, most commonly a norm of the parameter vector.
- $\lambda = 0$ — original problem; larger $\lambda$ — more bias, less variance. The **bias–variance tradeoff** made tunable.

Regularizing the weights (not the outputs) biases the model toward functions that change gently — the class of functions matters, not the parameter values, and small-norm parameters usually mean low-sensitivity outputs.

## L2 Regularization (Ridge / Weight Decay)

$$
R(\theta) = \tfrac12\|\theta\|_2^2 = \tfrac12\sum_j \theta_j^2
$$

(The $\tfrac12$ makes the gradient clean.)

**Gradient:**

$$
\nabla J_{\text{reg}} = \nabla J_{\text{data}} + \lambda\theta
\qquad\Longrightarrow\qquad
\theta \leftarrow \theta - \eta\nabla J_{\text{data}} - \eta\lambda\theta
$$

Regroup:

$$
\boxed{\;\theta \leftarrow (1 - \eta\lambda)\,\theta \;-\; \eta\,\nabla J_{\text{data}}\;}
$$

Every step first **shrinks all weights by the factor $(1 - \eta\lambda) \in (0,1)$** — the "weight decay" name — then applies the ordinary data gradient. The factor is exactly the multiplicative pattern from chapter 02's stability analysis, here *deliberately* used to pull parameters toward zero.

**Closed form for least squares (ridge regression).** Minimize $\|Xw - y\|^2 + \lambda\|w\|^2$ (sum-of-squares scaling; equivalent up to the constant $n$):

$$
\nabla_w = 2X^\top(Xw - y) + 2\lambda w = 0
\;\;\Longrightarrow\;\;
(X^\top X + \lambda I)\,w = X^\top y
\;\;\Longrightarrow\;\;
w^* = (X^\top X + \lambda I)^{-1} X^\top y
$$

Adding $\lambda I$ shifts the eigenvalues up by $\lambda$ — this both shrinks the solution toward zero and fixes the singularity when $X^\top X$ is rank-deficient (section 01: minimum-norm least squares was the other fix; ridge gives a unique answer always).

**Worked example (one feature).** Data $(x,y) = (1,1),\ (2,4)$ — in section 01's notation $X^\top X = 1^2 + 2^2 = 5$, $X^\top y = 1\cdot1 + 2\cdot4 = 9$.

| $\lambda$ | $w = 9/(5+\lambda)$ |
|---|---|
| 0 (plain) | $1.8$ |
| 1 | $1.5$ |
| 5 | $0.9$ |

As $\lambda$ grows, the coefficient shrinks toward $0$ — bias increases, variance decreases; at $\lambda \to \infty$, $w \to 0$ (constant-mean model: extreme underfit).

## L1 Regularization (Lasso / Sparsity)

$$
R(\theta) = \|\theta\|_1 = \sum_j |\theta_j|
$$

**Why it produces zeros.** The soft-thresholding fact — for the one-parameter problem

$$
\min_w\; \tfrac12(w - a)^2 + \lambda|w|
\qquad\Longrightarrow\qquad
w^* = \operatorname{sign}(a)\,\max(|a| - \lambda,\ 0)
$$

*Reason (subgradient):* for $w > 0$ the objective's slope is $(w - a) + \lambda$, zero at $w = a - \lambda$ — only if $a > \lambda$; for $w < 0$ symmetric at $w = a + \lambda$ — only if $a < -\lambda$; when $|a| \le \lambda$ the slope at $0$ points inward from both sides and $w^* = 0$. Compare L2's smooth $w^* = a/(1+\lambda)$, which never exactly vanishes — **L1 sets small coefficients to zero, L2 only shrinks them.**

Geometric reading: the L1 ball $|\theta_1| + |\theta_2| \le t$ is a diamond whose corners sit on the axes; level curves of the data term tend to touch the constraint first at a corner ⇒ axis-aligned (sparse) solutions. L2's round ball touches generically off-axis.

**When to prefer which.** L1 — you believe only a few features matter (feature selection, interpretability). L2 — many small correlated effects (stability, ridge's closed form). Elastic net combines both ($\lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2$).

## The Bayesian View (Section 04, Recall)

| penalty | prior | reading |
|---|---|---|
| $\lambda\|\theta\|^2/2$ | Gaussian $\theta \sim N(0, \tau^2 I)$, $\lambda = \sigma^2/\tau^2$ | shrink toward zero, variance $\tau^2$ |
| $\lambda\|\theta\|_1$ | Laplace $\theta_j \sim \text{Laplace}(0, b)$ | heavier tails, spikier at 0 → sparsity |
| $\lambda = 0$ | uniform (improper) | pure MLE |

Regularization and MAP estimation are the same operation viewed from optimization and probability. Choosing $\lambda$ ≈ choosing prior strength; selecting it on validation data ≈ empirical Bayes (honest note: that's what practitioners actually do).

## Beyond Penalties

- **Early stopping:** terminate training when validation error stops improving — fewer effective updates acts like an implicit $\lambda$ (gradient descent on the unregularized loss from a zero initialization biases toward small-norm solutions; the formal statement needs the convex-case path-norm analysis — we state the practice, not the theorem).
- **Noise injection:** input noise or dropout (section 08) regularize stochastically — same bias–variance logic, different mechanism.
- **Data augmentation** (flips, crops in vision) shrinks generalization error by expanding the effective training distribution — regularization through data, not objectives.

## Connection to AI

- **Weight decay** is on by default in most training recipes; forgetting it (or mis-scaling it with batch size) measurably changes results.
- **$\lambda$ tuning on validation sets** is hyperparameter search — the statistics discipline of held-out estimation applied to the regularizer itself.
- **Double descent** (modern observation): beyond a threshold, *more* parameters without regularization can reduce test error — the classical bias–variance curve is not the whole story; regularization interacts with overparameterization in ways still under study.
- **Language-model losses rarely carry explicit L1/L2** — but the ideas live on: weight decay variants, dropout, and the implicit regularization of training dynamics (chapter 04's noise) all control the same variance.
- **Feature selection via L1** predates deep learning and remains standard in sparse/statistical modeling.

## Common Mistakes

- Forgetting that $R$ must be scaled consistently with the loss: averaging the loss but summing the penalty (or vice versa) silently changes $\lambda$'s meaning when $n$ or batch size changes.
- Regularizing the **bias/intercept** — convention says don't (we only want to shrink the function's sensitivity to inputs, not its baseline).
- Tuning $\lambda$ on the training set — the penalty strength must be chosen on data not used to fit.
- Expecting L1 to be "just L2 but sparser" — their geometry differs; solutions are genuinely different, not ordered.
- Assuming more regularization is safer — $\lambda$ too large underfits: high bias, both train and test error rise (worked table above: $\lambda = 5$ halves the coefficient).
- Applying regularization twice — e.g., MAP loss plus the same explicit penalty (double counting the prior).

## Exercises

1. Write the L2-regularized objective for squared loss on $n$ examples with parameters $(w, b)$, penalizing $w$ only with strength $\lambda$: $R(w) = \tfrac12 w^2$. Give $\partial J_{\text{reg}}/\partial w$.
2. In the weight-decay form $\theta \leftarrow (1-\eta\lambda)\theta - \eta\nabla J_{\text{data}}$, what happens to a fixed $\theta$ as steps accumulate with no data gradient? (Write $\theta_k$.)
3. Ridge, one feature: $X^\top X = 10$, $X^\top y = 20$. Find $w$ for $\lambda = 0$ and $\lambda = 5$.
4. Apply soft-thresholding: $\tfrac12(w - a)^2 + 3|w|$ for $a = 1$, $a = 5$, and $a = -2$.
5. Which prior corresponds to L1 — Gaussian or Laplace — and why does that shape produce sparsity? (One sentence each.)
6. You increase $\lambda$ and find *both* training and validation error rising. What is this regime called, and which way should you move $\lambda$?
