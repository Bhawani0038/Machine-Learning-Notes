# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 15: Binomial Distribution

### What We Will Cover
- What is Binomial Distribution?
- Conditions for Binomial Distribution
- Bernoulli vs Binomial
- Probability Formula
- Simple Example
- Mean & Variance
- Real-Life Example
- Python Implementation
- ML Connection
- Key Points

---

## 1. What is Binomial Distribution?

**Binomial Distribution gives the probability of getting a certain number of successes in a fixed number of independent trials.**

In simple words:

> **Multiple Bernoulli trials → Count the number of successes.**

Example:

> Out of 10 students, what is the probability that exactly 7 pass?

---

# 2. Conditions for Binomial Distribution

A situation follows a binomial distribution when:

### 1. Fixed number of trials

There must be a fixed number:

$$
n
$$

Example:

```text
10 students
```

---

### 2. Two possible outcomes

Each trial has:

```text
Success / Failure
```

Example:

```text
Pass / Fail
```

---

### 3. Trials are independent

The outcome of one trial does not affect another.

---

### 4. Probability of success remains constant

The probability of success is:

$$
p
$$

for every trial.

---

# 3. Bernoulli vs Binomial

| Bernoulli | Binomial |
|---|---|
| One trial | Multiple trials |
| Two outcomes | Two outcomes per trial |
| Success/Failure | Number of successes |
| Example: one student passes | Example: 10 students pass |
| $n = 1$ | $n > 1$ |

Remember:

$$
\boxed{\text{Binomial} = \text{Repeated Bernoulli Trials}}
$$

---

# 4. Binomial Probability Formula

The probability of getting exactly $k$ successes in $n$ trials is:

$$
\boxed{
P(X=k)=
\binom{n}{k}
p^k(1-p)^{n-k}
}
$$

Where:

- $n$ = number of trials
- $k$ = number of successes
- $p$ = probability of success
- $1-p$ = probability of failure
- $\binom{n}{k}$ = number of ways to choose $k$ successes from $n$ trials

---

# 5. Simple Example

Suppose:

- Number of students = 5
- Probability of passing = 0.8

Question:

> What is the probability that exactly 4 students pass?

Given:

$$
n=5
$$

$$
k=4
$$

$$
p=0.8
$$

$$
1-p=0.2
$$

Formula:

$$
P(X=4)=
\binom{5}{4}(0.8)^4(0.2)^1
$$

Now:

$$
\binom{5}{4}=5
$$

Therefore:

$$
P(X=4)=
5(0.8)^4(0.2)
$$

$$
=5(0.4096)(0.2)
$$

$$
=0.4096
$$

Therefore:

$$
\boxed{P(X=4)=40.96\%}
$$

---

# 6. Mean of Binomial Distribution

The expected number of successes is:

$$
\boxed{E(X)=np}
$$

Example:

$$
n=5,\quad p=0.8
$$

Therefore:

$$
E(X)=5(0.8)
$$

$$
=4
$$

So we expect **4 successes on average**.

---

# 7. Variance

Binomial variance:

$$
\boxed{\operatorname{Var}(X)=np(1-p)}
$$

For:

$$
n=5,\quad p=0.8
$$

$$
\operatorname{Var}(X)=5(0.8)(0.2)
$$

$$
=0.8
$$

Standard deviation:

$$
\operatorname{SD}(X)=\sqrt{0.8}
$$

$$
\approx0.894
$$

---

# 8. Real-Life Example — Customer Conversion

Suppose an e-commerce website has:

```text
100 visitors
```

Historical conversion probability:

$$
p=0.05
$$

We can model the number of customers who purchase using:

$$
X\sim\operatorname{Binomial}(100,0.05)
$$

The expected number of purchases is:

$$
E(X)=np
$$

$$
=100(0.05)
$$

$$
=5
$$

So the expected number of purchases is:

**5 customers.**

---

# 9. Python Implementation

Using NumPy:

```python
import numpy as np

n = 5
p = 0.8

probability = np.random.binomial(
    n=5,
    p=0.8,
    size=10000
)

print(np.mean(probability))
```

The average will be approximately:

```text
4
```

because:

$$
E(X)=np=5\times0.8=4
$$

---

## Exact Probability with Python

```python
from math import comb

n = 5
k = 4
p = 0.8

probability = (
    comb(n, k)
    * p**k
    * (1-p)**(n-k)
)

print(probability)
```

Output:

```text
0.4096
```

Therefore:

```text
40.96%
```

---

# 10. Machine Learning Connection

Binomial distribution is useful when we are interested in the **number of successes across repeated binary outcomes**.

Examples:

- Number of customers who convert
- Number of emails classified as spam
- Number of students who pass
- Number of successful loan approvals
- Number of defective products

It also provides important statistical intuition for **binary outcomes and classification data**.

---

## Key Points

- Binomial distribution counts **successes in a fixed number of trials**.
- It requires:
  - Fixed number of trials
  - Two possible outcomes
  - Independent trials
  - Constant success probability
- Formula:

$$
P(X=k)=
\binom{n}{k}p^k(1-p)^{n-k}
$$

- Mean:

$$
E(X)=np
$$

- Variance:

$$
\operatorname{Var}(X)=np(1-p)
$$

- **Bernoulli = one trial**
- **Binomial = multiple Bernoulli trials**
- Common applications include **conversion, pass/fail, success/failure, and binary outcome analysis**.