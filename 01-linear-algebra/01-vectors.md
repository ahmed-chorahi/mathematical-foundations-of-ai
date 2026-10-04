# Vectors

## Why It Matters for AI

Every piece of data a model sees arrives as a list of numbers: a pixel grid flattened into a vector, a sentence mapped to an embedding, a set of measurements called a feature vector. Training turns parameters into vectors too. Vectors are the atomic data structure of AI — this chapter defines them and the two operations (addition, scalar multiplication) from which most of linear algebra grows.

## Scalars, Vectors, Matrices

- A **scalar** is a single number: $3$, $-0.5$, $\pi$.
- A **vector** is an ordered list of numbers: $\mathbf{x} = (x_1, x_2, \ldots, x_n)$.
- A **matrix** is a rectangular grid of numbers (chapter 03).

A scalar is a vector with one component; a vector is a matrix with one column. Keeping this in mind makes later generalizations painless.

## Vector Notation

A column vector of length $n$:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n
$$

$\mathbb{R}^n$ is the set of all real $n$-component vectors. Component $i$ is written $x_i$ (subscript = position, not a power — the convention from section 00).

Row vectors are written $\mathbf{x}^\top$ (transpose, chapter 03):

$$
\mathbf{x}^\top = \begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix}
$$

Three equivalent readings of $\mathbf{x} = (x_1, x_2, x_3) = (1, 2, 3)$:

1. a **point** in 3-dimensional space (coordinates),
2. a **list of features** of one example (height, weight, age),
3. an **arrow** from the origin to that point.

Same object, three intuitions — geometry, data, and physics all use the notation.

## Vector Addition

Add componentwise:

$$
\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix}
= \begin{bmatrix} 1 + 3 \\ 2 + 4 \end{bmatrix}
= \begin{bmatrix} 4 \\ 6 \end{bmatrix}
$$

**Requirements:** both vectors must have the same dimension. The sum $\mathbf{x} + \mathbf{y}$ exists only in $\mathbb{R}^n + \mathbb{R}^n$.

Properties: commutative ($\mathbf{x}+\mathbf{y} = \mathbf{y}+\mathbf{x}$), associative, and there is a **zero vector** $\mathbf{0} = (0,\ldots,0)$ with $\mathbf{x} + \mathbf{0} = \mathbf{x}$.

Geometric reading: the sum is the diagonal of the parallelogram formed by the two arrows — two displacements applied in sequence.

## Scalar Multiplication

Multiply every component by the scalar:

$$
3 \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 3 \\ -6 \end{bmatrix}
$$

Properties: $c(\mathbf{x} + \mathbf{y}) = c\mathbf{x} + c\mathbf{y}$, $(c+d)\mathbf{x} = c\mathbf{x} + d\mathbf{x}$, $(cd)\mathbf{x} = c(d\mathbf{x})$.

Geometric reading: scaling stretches or shrinks the arrow; $c < 0$ also flips its direction. A scalar multiple $c\mathbf{x}$ always stays on the line through the origin and $\mathbf{x}$ — this line is the **span** of $\mathbf{x}$, a concept expanded in chapter 04.

## Linear Combinations

Adding scaled vectors:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k
$$

is a **linear combination** of $\mathbf{v}_1,\ldots,\mathbf{v}_k$ with coefficients $c_1,\ldots,c_k$.

This single concept carries the section: the columns of a matrix multiplied by a vector are a linear combination of the columns (chapter 03); the set of all linear combinations of some vectors is their **span** (chapter 04); fitting a model is choosing combination weights.

**Worked example.** With $\mathbf{v}_1 = (1, 0)$, $\mathbf{v}_2 = (0, 1)$:

$$
2\mathbf{v}_1 - 3\mathbf{v}_2 = (2, 0) + (0, -3) = (2, -3)
$$

Any point in the plane is reachable — $\mathbf{v}_1, \mathbf{v}_2$ span $\mathbb{R}^2$.

## Equality and Special Vectors

$\mathbf{x} = \mathbf{y}$ iff every component matches: $x_i = y_i$ for all $i$.

- **Zero vector** $\mathbf{0}$: all components zero.
- **Standard basis vectors**: $\mathbf{e}_1 = (1, 0, \ldots, 0)$, $\mathbf{e}_2 = (0, 1, 0, \ldots)$, … — one 1 in each position.
- **One-hot vector**: a unit vector with a single 1 and 0s elsewhere — the encoding of a single category (one-hot labels for classification, token id embeddings lookup).

**Worked example (one-hot).** A 4-word vocabulary, word 3: $(0, 0, 1, 0)$. Classification layers often end with one component per class, and the index of the maximum component is the predicted class.

## Example: A Feature Vector

Three measurements of one house: size = 120 m², age = 10 years, rooms = 4.

$$
\mathbf{x} = \begin{bmatrix} 120 \\ 10 \\ 4 \end{bmatrix} \in \mathbb{R}^3
$$

A dataset of $n$ houses is $n$ such vectors, stacked into a matrix (chapter 03). A model's prediction is a function of this vector: $\hat{y} = f(\mathbf{x})$.

## Connection to AI

- **Embeddings are vectors.** A token, image, or user is mapped to $\mathbf{x} \in \mathbb{R}^d$; the coordinates are learned, not given. All downstream reasoning (similarity, attention) operates on these vectors.
- **Parameters are vectors and matrices.** The weights of a layer form a matrix $W$; all parameters of a network can be stacked into one giant vector $\theta$ — gradient descent then updates that vector (sections 05, 08).
- **Addition and scaling are the cheapest operations in a network.** Bias addition $\mathbf{z} + \mathbf{b}$ and weight scaling happen in every layer.
- **Data lives in $\mathbb{R}^n$.** Distance between vectors = similarity between examples; the geometry of that space decides what a model can learn.

## Common Mistakes

- Adding vectors of different lengths — undefined.
- Writing $x_i$ to mean $x^i$. Subscript = index; exponent = power.
- Treating $(x_1, x_2)$ as a scalar expression instead of a single object with two components.
- Confusing the vector with its components: $\mathbf{x} \in \mathbb{R}^3$ is one object; $x_1$ is one number.

## Exercises

1. Compute $\mathbf{u} + \mathbf{v}$ and $2\mathbf{u} - 3\mathbf{v}$ for $\mathbf{u} = (1, 2, 3)$, $\mathbf{v} = (4, -1, 0)$.
2. Write the standard basis vectors of $\mathbb{R}^3$ and the one-hot vector for class 2 of 4 classes.
3. Express $(7, -1)$ as a linear combination of $\mathbf{e}_1 = (1,0)$ and $\mathbf{e}_2 = (0,1)$.
4. Show that $\mathbf{x} + \mathbf{y} = \mathbf{y} + \mathbf{x}$ componentwise for arbitrary $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$.
5. A layer computes $\mathbf{z} = 2\mathbf{x} + \mathbf{b}$ with $\mathbf{x} = (1, 3)$ and $\mathbf{b} = (-1, 5)$. Find $\mathbf{z}$.
6. In words: what is the set of all vectors $c\mathbf{v}$ for a fixed $\mathbf{v} \in \mathbb{R}^2$ with $\mathbf{v} \ne \mathbf{0}$?
