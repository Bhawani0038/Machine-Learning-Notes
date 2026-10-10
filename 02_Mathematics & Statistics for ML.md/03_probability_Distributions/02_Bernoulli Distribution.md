# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 14: Bernoulli Distribution

### What We Will Cover
- What is Bernoulli Distribution?
- Binary Outcomes
- Success & Failure
- Probability Formula
- Mean and Variance
- Real-Life Examples
- Bernoulli vs Binomial
- Python Implementation
- ML Connection
- Key Points

---

## 1. What is Bernoulli Distribution?

**Bernoulli Distribution describes an experiment that has exactly two possible outcomes.**

Usually represented as:

```text
Success = 1
Failure = 0
```

Examples:

- Pass / Fail
- Yes / No
- True / False
- Spam / Not Spam
- Disease / No Disease
- Churn / No Churn
- Click / No Click

---

## 2. Example — Student Result

Suppose:

```text
Pass = 1
Fail = 0
```

If the probability of passing is:

$$
P(\text{Pass})=0.8
$$

then:

$$
P(\text{Fail})=1-0.8=0.2
$$

The random variable $X$ can be:

| X | Meaning | Probability |
|---:|---|---:|
| 0 | Fail | 0.2 |
| 1 | Pass | 0.8 |

This is a **Bernoulli distribution**.

---

# 3. Bernoulli Probability Formula

Let:

$$
X\sim\operatorname{Bernoulli}(p)
$$

where:

$$
p=P(X=1)
$$

Then:

$$
P(X=x)=p^x(1-p)^{1-x}
$$

where:

$$
x\in\{0,1\}
$$

### For Success: $x=1$

$$
P(X=1)=p
$$

### For Failure: $x=0$

$$
P(X=0)=1-p
$$

---

# 4. Mean of Bernoulli Distribution

The expected value or mean is:

$$
\boxed{E(X)=p}
$$

If:

$$
p=0.8
$$

then:

$$
E(X)=0.8
$$

Important:

The mean doesn't mean that an individual student gets **0.8 pass**.

It represents the **long-run average** of the binary variable.

For example, over many students:

```text
1, 1, 0, 1, 1, 1, 0, ...
```

the average can approach `0.8` if the true probability of success is 0.8.

---

# 5. Variance

For Bernoulli distribution:

$$
\boxed{\operatorname{Var}(X)=p(1-p)}
$$

If:

$$
p=0.8
$$

then:

$$
\operatorname{Var}(X)=0.8(1-0.8)
$$

$$
=0.8\times0.2
$$

$$
=0.16
$$

Standard deviation:

$$
\operatorname{SD}(X)=\sqrt{0.16}=0.4
$$

---

# 6. Real-Life Example — Customer Churn

Suppose:

```text
Churn = 1
No Churn = 0
```

Historical data suggests:

$$
P(\text{Churn})=0.2
$$

Therefore:

$$
P(\text{No Churn})=0.8
$$

| Outcome | Value | Probability |
|---|---:|---:|
| No Churn | 0 | 0.8 |
| Churn | 1 | 0.2 |

This is Bernoulli because there are exactly **two outcomes**.

---

# 7. Bernoulli Distribution in ML

Many classification problems have a binary target:

```text
        Target
          ↓
     ┌────┴────┐
     ↓         ↓
     0         1
     ↓         ↓
    No        Yes
```

Examples:

| Problem | 0 | 1 |
|---|---|---|
| Student | Fail | Pass |
| Customer | No Churn | Churn |
| Email | Not Spam | Spam |
| Loan | Rejected | Approved |
| Disease | Healthy | Diseased |

This is closely related to **binary classification**.

---

# 8. Bernoulli vs Binomial

This is an important distinction.

### Bernoulli

**One trial**

Example:

> Will one student pass?

```text
Pass / Fail
```

### Binomial

**Multiple independent Bernoulli trials**

Example:

> How many students out of 10 will pass?

```text
10 students
↓
10 Bernoulli trials
↓
Number of successes
```

So:

$$
\boxed{\text{Binomial} = \text{Multiple Bernoulli Trials}}
$$

---

# 9. Python Implementation

Using NumPy:

```python
import numpy as np

p = 0.8

outcomes = np.random.binomial(
    n=1,
    p=p,
    size=10
)

print(outcomes)
```

Possible output:

```text
[1 1 0 1 1 1 0 1 1 0]
```

Here:

```text
1 → Success
0 → Failure
```

---

## Key Points

- Bernoulli Distribution has **exactly two outcomes**.
- Usually represented as:
  - `1` → Success
  - `0` → Failure
- Probability of success = $p$.
- Probability of failure:

$$
1-p
$$

- Mean:

$$
E(X)=p
$$

- Variance:

$$
\operatorname{Var}(X)=p(1-p)
$$

- Bernoulli represents **one binary trial**.
- Binomial represents **multiple Bernoulli trials**.
- Bernoulli is highly relevant to **binary classification and probability modeling**.