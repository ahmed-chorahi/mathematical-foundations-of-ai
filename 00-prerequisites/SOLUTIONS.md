# 00 — Prerequisites: Solutions

Reasoning is shown; routine arithmetic is kept short.

## 01-algebra.md

**1. Solve $4x - 7 = 2x + 5$.**
Subtract $2x$: $2x - 7 = 5$. Add $7$: $2x = 12$. Divide by 2: $x = 6$.
Check: $4(6) - 7 = 17$ and $2(6) + 5 = 17$. ✓

**2. Solve $\frac{w}{3} + 2 = 8$.**
Subtract 2: $\frac{w}{3} = 6$. Multiply by 3: $w = 18$.
Check: $18/3 + 2 = 6 + 2 = 8$. ✓

**3. Solve $x^2 - 2x - 8 = 0$ by factoring.**
Find two numbers multiplying to $-8$ and adding to $-2$: $-4$ and $+2$.
$x^2 - 2x - 8 = (x - 4)(x + 2) = 0 \Rightarrow x = 4$ or $x = -2$.
Check: $16 - 8 - 8 = 0$ ✓ and $4 + 4 - 8 = 0$ ✓

**4. Solve $5 - 3x \le 2$.**
Subtract 5: $-3x \le -3$. Divide by $-3$ **and reverse the inequality**: $x \ge 1$.
Check with $x = 1$: $5 - 3 = 2 \le 2$ ✓; with $x = 0$ (excluded): $5 \not\le 2$ ✓.

**5. Rearrange $E = \frac{1}{2}mv^2$ for $v \ge 0$.**
Multiply by 2: $2E = mv^2$. Divide by $m$ (valid since $m > 0$): $v^2 = \frac{2E}{m}$. Take the non-negative root: $v = \sqrt{\dfrac{2E}{m}}$.

**6. Solve $|2x + 1| = 7$.**
Case 1: $2x + 1 = 7 \Rightarrow x = 3$. Case 2: $2x + 1 = -7 \Rightarrow x = -4$.
Check: $|2(3)+1| = 7$ ✓, $|2(-4)+1| = |-7| = 7$ ✓.

## 02-functions-and-graphs.md

**1. $f(x) = 3x - 2$, $g(x) = x^2 + 1$. Compute $f(2)$, $g(f(1))$, $f(g(1))$.**
$f(2) = 3(2) - 2 = 4$.
$f(1) = 1$, so $g(f(1)) = g(1) = 1^2 + 1 = 2$.
$g(1) = 2$, so $f(g(1)) = f(2) = 4$.
(Here they differ: $2 \ne 4$ — composition order matters.)

**2. Domain of $f(x) = \dfrac{1}{\sqrt{x}}$.**
Need $\sqrt{x}$ defined and non-zero: $x > 0$. Domain: $(0, \infty)$.

**3. Show $f(x) = 2x + 5$ is one-to-one and find $f^{-1}$.**
Suppose $f(a) = f(b)$: $2a + 5 = 2b + 5 \Rightarrow 2a = 2b \Rightarrow a = b$. So distinct inputs give distinct outputs: one-to-one.
Solve $y = 2x + 5$: $x = \dfrac{y - 5}{2}$. Thus $f^{-1}(y) = \dfrac{y-5}{2}$.
Check: $f(f^{-1}(11)) = f(3) = 11$ ✓

**4. $\lim_{x \to 2}(x^2 - 1)$.**
Polynomials are continuous, so the limit is direct substitution: $2^2 - 1 = 3$. No limit laws beyond continuity needed.

**5. ReLU$(-3)$ and ReLU$(-3 + 10^{-6})$.**
Both inputs are $\le 0$, so ReLU returns $0$ for both: $0$ and $0$.
(The function only changes branch at exactly $x = 0$.)

**6. Show $\sigma(-x) = 1 - \sigma(x)$ for $\sigma(x) = \frac{1}{1+e^{-x}}$.**

$$
1 - \sigma(x) = 1 - \frac{1}{1+e^{-x}} = \frac{1 + e^{-x} - 1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}}
$$

Divide numerator and denominator by $e^{-x}$: $= \dfrac{1}{e^{x} + 1} = \dfrac{1}{1+e^{-x}}\Big|_{-x} = \sigma(-x)$. ✓

## 03-exponents-and-logarithms.md

**1. Simplify $\dfrac{3^{2n}}{3^{n-1}}$.**
Quotient rule: $3^{2n - (n-1)} = 3^{2n - n + 1} = 3^{n+1}$.

**2. $\log_2 32$, $\log_{10} 1000$, $\ln 1$.**
$2^5 = 32 \Rightarrow \log_2 32 = 5$.
$10^3 = 1000 \Rightarrow \log_{10} 1000 = 3$.
$e^0 = 1 \Rightarrow \ln 1 = 0$.

**3. Expand $\ln\left(\dfrac{x^2 y}{z^3}\right)$.**
$$
\ln(x^2 y) - \ln(z^3) = \ln(x^2) + \ln y - \ln(z^3) = 2\ln x + \ln y - 3\ln z
$$

**4. Solve $e^{0.5t} = 4$.**
Take $\ln$ of both sides (one-to-one, so no roots are lost): $0.5t = \ln 4$, so $t = 2\ln 4$.
Using $\ln 4 \approx 1.386$: $t \approx 2.77$.

**5. Show $\log_a x \cdot \log_x a = 1$.**
Change of base: $\log_a x = \frac{\ln x}{\ln a}$ and $\log_x a = \frac{\ln a}{\ln x}$.
Product: $\frac{\ln x}{\ln a}\cdot\frac{\ln a}{\ln x} = 1$ (valid since $a, x \ne 1$ and all arguments positive).

**6. $p = 0.01$; compute $-\log_2 p$.**
$$
-\log_2(0.01) = \log_2(100) = \frac{\ln 100}{\ln 2} \approx \frac{4.605}{0.693} \approx 6.64
$$
Roughly 6.64 yes/no questions are needed, on average under an optimal coding, to identify an event with probability $1/100$ — the connection to entropy (section 06).

## 04-summation-and-notation.md

**1. $\sum_{i=1}^{5} i^2$.**
$1 + 4 + 9 + 16 + 25 = 55$.

**2. Simplify $\sum_{i=1}^{n}(3i - 2)$.**
$$
3\sum_{i=1}^{n} i - \sum_{i=1}^{n} 2 = 3\cdot\frac{n(n+1)}{2} - 2n = \frac{3n^2 + 3n - 4n}{2} = \frac{3n^2 - n}{2}
$$
Check for $n = 1$: $(3 - 1)/2 = 1$ ✓ (only term is $3\cdot1 - 2 = 1$).

**3. $A = \{a,b,c\}$, $B = \{b,c,d\}$.**
$A \cup B = \{a,b,c,d\}$, $A \cap B = \{b,c\}$, $A \setminus B = \{a\}$.

**4. De Morgan's law, $A = \{1,2\}$, $B = \{2,3\}$, $S = \{1,2,3,4\}$.**
$A \cup B = \{1,2,3\}$, so $(A\cup B)^c = \{4\}$.
$A^c = \{3,4\}$, $B^c = \{1,4\}$, so $A^c \cap B^c = \{4\}$. Equal ✓

**5. $\sum_{i=1}^{3}\sum_{j=1}^{2}(i+j)$.**
Inner sums: $i=1$: $(1+1)+(1+2) = 5$; $i=2$: $3 + 4 = 7$; $i=3$: $4 + 5 = 9$.
Total: $5 + 7 + 9 = 21$.

**6. Average squared error.**
$$
\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
This is the mean squared error — the loss function for linear regression (section 07).

## 05-geometry-and-trigonometry.md

**1. Distance $(0,0)$ to $(3,4)$.**
$\sqrt{3^2 + 4^2} = \sqrt{9+16} = \sqrt{25} = 5$.

**2. $\|\mathbf{x}\|$ for $(1,-2,2)$.**
$\sqrt{1^2 + (-2)^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3$.

**3. $60°$ in radians; $\sin$, $\cos$.**
$60 \times \pi/180 = \pi/3$. From the table: $\sin(\pi/3) = \sqrt{3}/2 \approx 0.866$, $\cos(\pi/3) = 1/2$.

**4. Pythagorean identity at $\theta = \pi/3$.**
$\sin^2 + \cos^2 = \left(\frac{\sqrt3}{2}\right)^2 + \left(\frac{1}{2}\right)^2 = \frac{3}{4} + \frac{1}{4} = 1$. ✓

**5. Law of cosines, sides 3 and 4, included angle $\pi/2$.**
$c^2 = 3^2 + 4^2 - 2\cdot3\cdot4\cdot\cos(\pi/2) = 9 + 16 - 0 = 25 \Rightarrow c = 5$.
($\cos(\pi/2) = 0$ recovers the Pythagorean theorem — a 3-4-5 triangle.)

**6. Unit vectors at angle $\pi/3$.**
$\mathbf{x}\cdot\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta = 1 \cdot 1 \cdot \cos(\pi/3) = \tfrac{1}{2}$.
