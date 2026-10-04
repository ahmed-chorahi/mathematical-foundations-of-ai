# Geometry and Trigonometry

## Why It Matters for AI

Vectors live in space, and the geometric language — length, distance, angle — is exactly what the algebraic operations compute. The dot product equals $ab\cos\theta$; cosine similarity is how similarity is measured in embedding space; the Pythagorean theorem *is* the Euclidean norm. This chapter states the small amount of plane geometry and trigonometry that later sections use.

## Coordinates and Distance

A point in the plane is a pair $(x, y)$. The distance between $P_1 = (x_1, y_1)$ and $P_2 = (x_2, y_2)$ follows from the Pythagorean theorem applied to the right triangle with legs $|x_2 - x_1|$ and $|y_2 - y_1|$:

$$
d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}
$$

In $n$ dimensions the same formula holds:

$$
d = \sqrt{\sum_{i=1}^{n}(x_{2,i} - x_{1,i})^2}
$$

**Worked example.** Distance between $(1, 2)$ and $(4, 6)$:

$$
d = \sqrt{(4-1)^2 + (6-2)^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

This is the **Euclidean distance**, the $L^2$ distance — the default notion of "nearness" in machine learning, and precisely the norm $\|\mathbf{x}\|$ of section 01.

## The Pythagorean Theorem

For a right triangle with legs $a, b$ and hypotenuse $c$:

$$
a^2 + b^2 = c^2
$$

Equivalently: the squared length of a vector equals the sum of the squares of its components. This identity is why squaring, summing, and taking a square root computes length, and why $\|\mathbf{x}\|^2 = \mathbf{x}^\top\mathbf{x}$ (section 01).

## Angles

An **angle** measures rotation between two directions. Two systems exist:

- **degrees**: a full turn = $360°$,
- **radians** (standard in mathematics): a full turn = $2\pi$ radians, so $180° = \pi$ rad.

Conversion: $\text{radians} = \text{degrees} \times \pi / 180$.

Right angle: $90° = \pi/2$ rad. Two vectors are **orthogonal** (perpendicular) when the angle between them is $\pi/2$ — the geometric fact behind "orthogonal vectors" in linear algebra.

**Reference triangle.** Place a point on a circle of radius $r$ at angle $\theta$ from the positive x-axis. Its coordinates are:

$$
x = r\cos\theta, \qquad y = r\sin\theta
$$

This one picture defines sine and cosine for all angles.

## Sine and Cosine

For a right triangle with angle $\theta$:

$$
\sin\theta = \frac{\text{opposite}}{\text{hypotenuse}}, \qquad
\cos\theta = \frac{\text{adjacent}}{\text{hypotenuse}}, \qquad
\tan\theta = \frac{\sin\theta}{\cos\theta}
$$

Values at key angles:

| $\theta$ | $0$ | $\pi/6\ (30°)$ | $\pi/4\ (45°)$ | $\pi/3\ (60°)$ | $\pi/2\ (90°)$ |
|---|---|---|---|---|---|
| $\sin\theta$ | $0$ | $1/2$ | $\sqrt{2}/2 \approx 0.707$ | $\sqrt{3}/2 \approx 0.866$ | $1$ |
| $\cos\theta$ | $1$ | $\sqrt{3}/2$ | $\sqrt{2}/2$ | $1/2$ | $0$ |

**Range:** $-1 \le \sin\theta \le 1$ and $-1 \le \cos\theta \le 1$ for all $\theta$.

### Identities used later

**Pythagorean identity** (from $x^2 + y^2 = r^2$ with $r = 1$):

$$
\sin^2\theta + \cos^2\theta = 1
$$

**Angle difference** (needed for the law of cosines below):

$$
\cos(\alpha - \beta) = \cos\alpha\cos\beta + \sin\alpha\sin\beta
$$

**Orthogonality:** $\cos(\pi/2) = 0$ — perpendicular directions have zero cosine.

**Worked example.** $\cos(\pi/4) = \sin(\pi/4) = \sqrt{2}/2$, consistent with the identity: $(\sqrt2/2)^2 + (\sqrt2/2)^2 = 1/2 + 1/2 = 1$ ✓

## The Law of Cosines

For a triangle with sides $a, b$ and included angle $\theta$ opposite side $c$:

$$
c^2 = a^2 + b^2 - 2ab\cos\theta
$$

This generalizes the Pythagorean theorem: when $\theta = 90°$, $\cos\theta = 0$ and $c^2 = a^2 + b^2$.

Why it is important here: applying the law of cosines to two vectors $\mathbf{x}, \mathbf{y}$ placed tail to tail, with $c^2 = \|\mathbf{x} - \mathbf{y}\|^2$, produces

$$
\|\mathbf{x}-\mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta
$$

Meanwhile expanding the left side algebraically gives $\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2(\mathbf{x}\cdot\mathbf{y})$. Comparing the two yields the fundamental identity (proved fully in section 01):

$$
\mathbf{x}\cdot\mathbf{y} = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta
$$

Geometry and algebra agree — the dot product measures angle.

## Connection to AI

- **Distance = embedding similarity.** Close embeddings in $\mathbb{R}^d$ mean small Euclidean distance; recommendation and search use exactly this geometry.
- **Cosine similarity = direction.** $\cos\theta$ between two embedding vectors ignores magnitude and compares direction — the standard similarity for text embeddings (section 01).
- **Orthogonality = independence of directions.** Orthogonal bases, orthogonal projections, and PCA's uncorrelated components all rest on $\cos(\pi/2) = 0$ (sections 01, 07).
- **Periodic functions.** Sinusoidal position encodings in transformers use $\sin$ and $\cos$ at geometrically increasing frequencies to inject position information (section 09).

## Common Mistakes

- Mixing degrees and radians: $\sin(30)$ in radians is not $0.5$ — it is $\sin(30 \text{ rad}) \approx -0.988$.
- Assuming $\sin(\alpha + \beta) = \sin\alpha + \sin\beta$ (it is false).
- Forgetting that $\sqrt{a^2 + b^2} \ne a + b$.
- Treating cosine similarity as distance: $\cos\theta$ near $1$ means *similar*; distance near $0$ means *similar* — the directions of "good" are opposite.

## Exercises

1. Find the distance between $(0, 0)$ and $(3, 4)$.
2. Compute $\|\mathbf{x}\|$ for $\mathbf{x} = (1, -2, 2)$ using the $n$-dimensional distance formula.
3. Convert $60°$ to radians, and give $\sin$ and $\cos$ at that angle from the table.
4. Verify the Pythagorean identity for $\theta = \pi/3$.
5. A triangle has sides 3 and 4 with included angle $\pi/2$. Find the third side with the law of cosines.
6. Two unit vectors have an angle $\pi/3$ between them. Using $\mathbf{x}\cdot\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$, compute their dot product.
