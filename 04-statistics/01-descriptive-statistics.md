# Descriptive Statistics

## Why It Matters for AI

Before a model touches data, the data gets summarized: means for normalization, variances for scaling, correlations for understanding feature relationships. These summaries are also the vocabulary for everything later — likelihoods estimate means and variances, and the bias–variance discussion (chapter 02) is built on the quantities defined here.

## Population vs Sample

- **Population**: the complete set of everything the statement is about — all users, all possible inputs. Parameters: $\mu$ (mean), $\sigma^2$ (variance).
- **Sample**: the $n$ observations actually collected. Statistics: $\bar{x}$ (sample mean), $s^2$ (sample variance).

You compute statistics from data; parameters are what you want to know but generally cannot observe. The gap between them is the subject of chapter 02. Convention: Greek ($\mu, \sigma$) for population, Latin ($\bar x, s$) for sample.

## Mean and Median

**Mean** (arithmetic average):

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

**Median**: the middle value when data are sorted (average of the two middles if $n$ even).

Worked example — dataset $[2, 4, 4, 4, 5, 5, 7, 9]$ ($n = 8$):

$$
\bar{x} = \frac{2+4+4+4+5+5+7+9}{8} = \frac{40}{8} = 5
$$

Sorted already; middle two values are $4$ and $5$, median $= 4.5$.

Mean uses every value but is pulled by outliers (add the value $100$ to the data and the mean rises from $5$ to $140/9 \approx 15.6$; the median moves only from $4.5$ to $5$). Median is robust; mean is differentiable and additive — which is why losses and gradients prefer means, while reporting often prefers medians.

## Variance and Standard Deviation

**Sample variance** — average squared deviation from the mean, with $n - 1$ in the denominator:

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

**Sample standard deviation:** $s = \sqrt{s^2}$ (same units as the data).

Why $n-1$ and not $n$: $\bar x$ is computed *from the same data*, which makes the deviations $x_i - \bar x$ sum to zero — they are slightly too small on average (you fit the center to the data). Dividing by $n-1$ corrects that bias exactly (derived in chapter 02). Dividing by $n$ gives $\hat\sigma^2_{\text{MLE}}$, biased low but the maximum-likelihood estimate — both appear in software; know which you are using.

**Worked example** on $[2, 4, 4, 4, 5, 5, 7, 9]$ with $\bar x = 5$:

| $x_i$ | $x_i - \bar x$ | $(x_i - \bar x)^2$ |
|---|---|---|
| 2 | −3 | 9 |
| 4 | −1 | 1 |
| 4 | −1 | 1 |
| 4 | −1 | 1 |
| 5 | 0 | 0 |
| 5 | 0 | 0 |
| 7 | 2 | 4 |
| 9 | 4 | 16 |

Sum of squares $= 32$.

$$
s^2 = \frac{32}{7} \approx 4.571, \qquad s \approx 2.138
$$

(population-style division: $32/8 = 4$, $s = 2$.)

Interpretation: typical point sits about $2.1$ units from the mean.

## Covariance and Correlation (Recap)

From section 03 — restated for data:

$$
s_{XY} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y),
\qquad
r = \frac{s_{XY}}{s_X s_Y} \in [-1, 1]
$$

$r$ (Pearson correlation) measures linear association in a unit-free scale: $r \approx 1$ strong positive linear relationship, $0$ none, $-1$ strong negative.

**Worked example.** $x = [1, 2, 3]$, $y = [2, 4, 6]$: $y = 2x$ exactly, so $r = 1$. $y = [1, 4, 9]$? (i.e., $y = x^2$) has high but not perfect linear correlation — correlation captures only the straight-line component.

## Standardization (Z-Scores)

$$
z_i = \frac{x_i - \bar{x}}{s}
$$

transforms data to mean 0, standard deviation 1. **Why AI does this:** features on different scales (age in years, income in dollars) distort distance-based and gradient-based methods — a $10$-unit income difference would swamp a $1$-unit age difference. Standardization puts features on equal footing before distances, dot products, or training. (A model with *one* feature can absorb scale into its weights; standardization matters when comparing across features or using methods sensitive to scale.)

## The Covariance Matrix of Data

For data matrix $X$ ($n$ rows, $d$ features, columns centered), the sample covariance matrix:

$$
S = \frac{1}{n-1} X^\top X
$$

entries $S_{jk}$ = covariance of features $j, k$ — section 01's Gram structure applied to centered features. $S$ is symmetric positive semi-definite (section 01), so its eigendecomposition is well-behaved: that matrix is PCA's input (section 07).

## Connection to AI

- **Normalization layers** standardize activations (mean 0, variance 1) per batch — literally this chapter's z-score, computed at training time (section 09's layer normalization).
- **Feature scaling** before training is standard preprocessing; unscaled inputs make loss surfaces elongated and gradient descent slow (section 05 shows the geometry).
- **Descriptive ≠ inferential.** Reporting "mean accuracy 94%" from a test set is a sample statistic; claiming the true rate is 94% needs chapter 05's uncertainty statements.
- **Correlation screening** identifies redundant features — linked to linear dependence (section 01) and PCA.

## Common Mistakes

- Using $1/n$ and $1/(n-1)$ interchangeably without noticing they differ (converge for large $n$, differ noticeably for small $n$).
- Reading correlation as causation — $r$ measures linear co-movement only.
- Reporting mean and std for heavily skewed data (income, loss values) where median and quartiles are more honest.
- Forgetting to standardize *consistently*: statistics computed on training data must be reused for test data (using test statistics leaks information).
- Assuming zero correlation means independence — the $Y = X^2$ counterexample from section 03.

## Exercises

1. Compute mean and median of $[3, 1, 4, 1, 5, 9, 2, 6]$.
2. Compute the sample variance of $[1, 3, 5]$ (mean first; then both $1/(n-1)$ and $1/n$ versions).
3. A dataset has mean $10$ and standard deviation $2$. Give the z-score of the value $14$ and say how many standard deviations from the mean it is.
4. Compute the correlation of $x = [1, 2, 3]$ with $y = [3, 2, 1]$ — sign only, with reasoning.
5. Why does dividing squared deviations by $n-1$ correct a bias? (One-sentence version.)
6. Two features have sample covariance matrix $\begin{bmatrix}4 & 0 \\ 0 & 9\end{bmatrix}$. What does the zero off-diagonal say, and what are the standard deviations?
