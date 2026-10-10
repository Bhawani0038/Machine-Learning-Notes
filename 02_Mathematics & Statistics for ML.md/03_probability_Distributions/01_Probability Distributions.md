# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 13: Probability Distributions

### What We Will Cover
- What is a Probability Distribution?
- Discrete Distribution
- Continuous Distribution
- Probability Mass Function (PMF)
- Probability Density Function (PDF)
- Probability Distribution Examples
- ML Applications
- Key Points

---

## 1. What is a Probability Distribution?

A **probability distribution describes how probabilities are distributed among possible values of a random variable.**

Example:

If we roll a dice:

```text
Outcome → 1  2  3  4  5  6
```

For a fair dice:

```text
Probability → 1/6 each
```

The distribution tells us how likely each outcome is.

---

# 2. Random Variable

A **random variable** is a variable whose value depends on the outcome of a random experiment.

Example:

Toss a coin.

Let:

$$
X =
\begin{cases}
1 & \text{if the outcome is Head}\\
0 & \text{if the outcome is Tail}
\end{cases}
$$

Here, `X` is a random variable.

---

# 3. Two Main Types

Probability distributions are broadly divided into:

```text
Probability Distributions
        │
        ├── Discrete
        │
        └── Continuous
```

---

# 4. Discrete Probability Distribution

A **discrete random variable** takes countable values.

Examples:

- Number of customers
- Number of students
- Number of defective products
- Dice result
- Number of heads
- Number of purchases

Example:

```text
Number of customers = 0, 1, 2, 3, 4, ...
```

Individual values have probabilities.

---

## PMF — Probability Mass Function

For a discrete random variable, we use a **Probability Mass Function (PMF)**.

It gives:

$$
P(X=x)
$$

Example:

For a fair dice:

| X | P(X) |
|---:|---:|
| 1 | 1/6 |
| 2 | 1/6 |
| 3 | 1/6 |
| 4 | 1/6 |
| 5 | 1/6 |
| 6 | 1/6 |

All probabilities must add up to:

$$
\sum P(X=x)=1
$$

---

# 5. Continuous Probability Distribution

A **continuous random variable** can take any value within a range.

Examples:

- Height
- Weight
- Temperature
- Salary
- House price
- Delivery time
- Blood pressure

For example:

```text
Height = 170 cm
Height = 170.5 cm
Height = 170.52 cm
Height = 170.523 cm
...
```

There can be infinitely many possible values.

---

# 6. PDF — Probability Density Function

For continuous variables, we generally use a **Probability Density Function (PDF)**.

Important point:

> For a continuous variable, the probability of one exact point is theoretically zero.

Instead, we calculate probability over an **interval**.

For example:

$$
P(170<X<180)
$$

means:

> Probability that height lies between 170 cm and 180 cm.

---

# 7. PMF vs PDF

| PMF | PDF |
|---|---|
| Discrete variables | Continuous variables |
| Countable values | Infinitely many possible values |
| Probability at a value | Density over a range |
| Dice result | Height |
| Number of customers | Temperature |
| Number of defects | House price |

---

# 8. Common Probability Distributions

### Discrete

- Bernoulli Distribution
- Binomial Distribution
- Poisson Distribution

### Continuous

- Normal Distribution
- Uniform Distribution
- Exponential Distribution

For ML, some particularly important distributions are:

```text
Bernoulli
Binomial
Normal
Uniform
Poisson
Exponential
```

---

# 9. Why Probability Distributions Matter in ML

Probability distributions help us understand:

### Data Shape

Example:

```text
Are most values around the average?
```

### Uncertainty

```text
How likely is an outcome?
```

### Outliers

```text
Is this value extremely unlikely?
```

### Statistical Modeling

Many statistical and ML methods make assumptions about how data is distributed.

For example, the **Normal Distribution** is very important in statistics and machine learning.

---

# 10. Real-Life Example — House Prices

Suppose house prices are:

```text
30L
35L
38L
40L
42L
45L
48L
50L
...
```

We can study the distribution to understand:

- Typical house price
- Spread
- Skewness
- Unusual prices
- Probability of prices falling within a range

This helps during **EDA and preprocessing**.

---

# 11. Important Distinction

A distribution is not simply a list of probabilities.

It describes the **overall behavior of a random variable**.

For example:

```text
Data
 ↓
Possible Values
 ↓
Frequency / Probability
 ↓
Distribution
```

---

## Key Points

- A **probability distribution** describes how probabilities are assigned to possible values.
- A **random variable** represents an uncertain numerical outcome.
- **Discrete** variables have countable values.
- **Continuous** variables can take infinitely many values within a range.
- **PMF** is used for discrete variables.
- **PDF** is used for continuous variables.
- Common distributions:
  - Bernoulli
  - Binomial
  - Poisson
  - Normal
  - Uniform
  - Exponential
- Probability distributions are important for **statistics, EDA, uncertainty, outlier detection, and ML modeling**.