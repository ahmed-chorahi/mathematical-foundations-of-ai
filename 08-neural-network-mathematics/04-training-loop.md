# The Training Loop

## Why It Matters for AI

Gradients are necessary but not sufficient: a network is trained by a *procedure* — sample data, forward, backward, update, repeat — with choices (learning rate, batch size, epochs, initialization, regularization) that determine whether it learns anything. This chapter is the engineering of section 05's SGD, applied to the object from chapters 02–03, plus the failure modes every practitioner meets.

## The Loop

For each epoch, for each minibatch:

1. **Forward:** compute $\hat y$ and $L$ (chapter 02).
2. **Backward:** compute $\nabla_\theta L$ for the batch (chapter 03).
3. **Update:** $\theta \leftarrow \theta - \eta \cdot \nabla_\theta \bar{L}_{\text{batch}}$ (section 05).
4. **Zero** the gradients (start next iteration clean).

Repeat for $E$ **epochs** (full passes over the data — section 05's unit of budget). The loop *is* section 05 chapter 04's SGD with $\theta = $ all $P$ parameters and $\nabla$ supplied by backprop — the two chapters are one algorithm.

**A minimal pure-Python skeleton** (one layer, one step — the pattern scales):

```python
def train_step(w, b, x, y, eta):
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    y_hat = 1 / (1 + __import__('math').exp(-z))
    L = -(__import__('math').log(y_hat + 1e-12) if y else
          __import__('math').log(1 - y_hat + 1e-12))
    d = y_hat - y                       # delta for sigmoid + BCE
    w = [wi - eta * d * xi for wi, xi in zip(w, x)]
    b = b - eta * d
    return w, b, L
```

Every framework's `loss.backward(); optimizer.step(); optimizer.zero_grad()` is this function with autodiff instead of hand-written deltas.

## What the Loss Curve Should Do

- **Training loss decreases steadily** — the primary health signal. Plateau ⇒ lower $\eta$, more capacity, or a broken gradient (chapter 03's gradient check).
- **Validation loss tracks down, then may rise** — the overfitting signature (section 04's bias–variance, section 05's regularization). Stop at the minimum (**early stopping** — a regularizer from section 05's list).
- **Both flat from the start** — learning-rate too small, features unscaled (section 04), or a dead/saturated activation (chapter 01).
- **Loss explodes to `NaN`** — $\eta$ too large (chapter 02 of section 05: divergence), exploding gradients, or $\log 0$ (clip probabilities / use stable loss forms).

## Initialization: Why Not Zeros?

Symmetric initialization breaks learning: if all weights of a layer start equal, every neuron in that layer computes the *same* function, receives the *same* gradient, and stays identical forever — width buys nothing (the symmetry must be broken to differentiate neurons). Standard practice: small **random** values (scale by layer size — the precise scaling keeps signal variance ≈ 1 through depth, the same variance-preservation logic as section 04's expectation bookkeeping). Biases typically start at $0$.

## Regularization and Data (Section 05, Deployed)

- Weight decay = section 05's penalty inside the optimizer — on by default in practice.
- Dropout: randomly zero a fraction of activations *during training* (scaled at test time) — noise as regularization (section 05's list; the stochasticity also approximates an ensemble).
- Data augmentation: more effective than almost any penalty — section 05's "through data" entry.
- **Monitor validation, never trust training loss alone** — the estimation discipline of section 04, now about $\theta$'s generalization rather than an estimator's bias.

## The Gradient Health Problem (Scale)

Backprop multiplies layer-to-layer (chapter 03): $|\delta^{(\ell)}| \sim \prod_{k>\ell} \|W^{(k)}\| \cdot f'$. Two regimes:

- **Vanishing:** products $\ll 1$ (small $f'$, small weights) — deep layers barely train. Chapter 01's sigmoid arithmetic ($0.25^L$).
- **Exploding:** products $\gg 1$ — updates overshoot, loss spikes. Fix: gradient clipping (cap update norm — the trust-region idea from section 05), careful init, normalization.

Residual connections (chapter 01 of section 09's neighborhood): $\mathbf{a}^{(\ell+1)} = f(\cdot) + \mathbf{a}^{(\ell)}$ — the identity path carries gradients with factor exactly $1$, no matter the depth; this single trick made hundred-layer networks trainable (stated as the architectural fact it is; proof-level analysis is research territory).

## Honest Statement of What Training Guarantees

- The objective of deep networks is **non-convex** (section 05): backprop delivers exact gradients; SGD converges to *a* critical point, not a provable global optimum.
- Practice: with overparameterization, most found minima have similar training loss; generalization depends on implicit biases of the algorithm (initialization, learning-rate schedule, noise) — active research, no complete theorem.
- **Therefore:** validation metrics, not training loss, are the arbiter. The mathematics guarantees *correct gradients and principled updates*; the *outcome* is empirical.

## Connection to AI

- **This loop trains everything after it:** CNNs, GANs, transformers, LLMs — same four steps; only the architecture (forward) and loss change.
- **Transformers' training additions** (section 10): cross-entropy over vocabulary per position (section 06/07 machinery), much larger $\theta$, learning-rate warmup and decay — schedule refinements, not new mechanics.
- **Hyperparameters** ($\eta$, batch size, weight decay, epochs) are selected by validation sweeps — optimization meets statistics' held-out discipline (section 04) at industrial scale.
- **Fine-tuning** is this loop started from pretrained $\theta_0$ with small $\eta$ — the same equations, different initialization (section 10's orbit, KL-constrained variants).
- **Reproducibility:** fixed seeds for init, batch order, and dropout — randomness enters at three points named above; control them or results drift.

## Common Mistakes

- Not zeroing gradients → accidental accumulation (or *thinking* you zeroed when the optimizer holds internal state).
- Training loss only — no validation curve: overfitting goes unseen (section 04's optimism bias).
- Raw-scale features + default $\eta$ → slow or divergent training — standardize first (section 04).
- Forgetting `eval()`-mode behaviors at test time (dropout off, normalization frozen) — train/test mismatch.
- Comparing losses across runs with different batch sizes at fixed $\eta$ without adjusting — effective step per example changed (section 05).
- Treating `NaN` as a framework bug before checking learning rate and input data (most often it's one of those two).
- Freezing layers but not their optimizer state (or vice versa) — stale momentum/second moments applied to dead weights.

## Exercises

1. Write the loop pseudocode for 3 epochs over 5 minibatches with explicit forward/backward/update/zero ordering.
2. A model overfits: training loss $0.12$ falling, validation loss $0.9$ rising for epochs. Name three interventions from this chapter and section 05.
3. Why does zero-initialized *weights* in every layer collapse all neurons of a layer into one function? (One sentence.)
4. Compute the surviving gradient factor after 8 sigmoid layers in the worst case; compare with 8 ReLU layers on active paths.
5. Name the three places randomness enters training as described in this chapter, and the fix for reproducibility.
6. State what backprop guarantees and what it does not, in one sentence each.
