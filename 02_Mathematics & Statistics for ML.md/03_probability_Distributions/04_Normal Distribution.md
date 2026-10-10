# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 16: Normal Distribution

### What We Will Cover
- What is Normal Distribution?
- Bell Curve
- Mean, Median & Mode
- Standard Deviation
- Shape of Normal Distribution
- 68–95–99.7 Rule
- Standard Normal Distribution
- Real-Life Examples
- ML Applications
- Python
- Key Points

---

## 1. What is Normal Distribution?

The **Normal Distribution** is one of the most important probability distributions in statistics and machine learning.

It is a continuous probability distribution with a characteristic **bell-shaped curve**.

```text
                    ▲
                  ███
                ███████
              ███████████
            ███████████████
         █████████████████████
───────█████████████████████████───────→ X
                    μ
```

Examples of measurements that can often be approximately normal in suitable populations:

- Height
- Measurement errors
- Test scores
- Blood pressure
- Manufacturing measurements

Not every real-world dataset is normally distributed.

---

# 2. Main Characteristics

A normal distribution is:

- Bell-shaped
- Symmetric around the mean
- Continuous
- Unimodal
- Defined by **mean** and **standard deviation**

For a perfectly normal distribution:

$$
\text{Mean} = \text{Median} = \text{Mode}
$$

---

# 3. Role of Mean

The mean determines the **center** of the distribution.

Example:

$$
\mu=50
$$

The distribution is centered around:

```text
50
```

If the mean changes:

```text
μ = 30
```

the entire curve shifts toward 30.

---

# 4. Role of Standard Deviation

Standard deviation controls the **spread** of the distribution.

### Small Standard Deviation

Values are concentrated around the mean.

```text
             ███
           ███████
         ███████████
────────██████████████────────
             μ
```

### Large Standard Deviation

Values are more spread out.

```text
       █████████████
     █████████████████
   █████████████████████
─────────────μ─────────────
```

So:

> **Mean controls location; standard deviation controls spread.**

---

# 5. Probability Density Function

The PDF of a normal distribution is:

$$
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

Where:

- $\mu$ = mean
- $\sigma$ = standard deviation
- $x$ = observed value

You don't need to memorize the formula initially.

The important intuition is:

$$
\boxed{\text{Normal distribution is determined by } \mu \text{ and } \sigma}
$$

---

# 6. 68–95–99.7 Rule

This is one of the most important properties of a normal distribution.

Approximately:

### Within 1 Standard Deviation

$$
\mu \pm \sigma
$$

contains about:

$$
68\%
$$

of observations.

### Within 2 Standard Deviations

$$
\mu \pm 2\sigma
$$

contains about:

$$
95\%
$$

### Within 3 Standard Deviations

$$
\mu \pm 3\sigma
$$

contains about:

$$
99.7\%
$$

---

## Visual Intuition

```text
                 68%
             ┌────────┐
             │        │
        95%  │        │
     ┌───────┴────────┴───────┐
     │                        │
99.7%└────────────────────────┘
    μ-3σ  μ-2σ  μ-σ  μ  μ+σ  μ+2σ  μ+3σ
```

This is also called the **Empirical Rule**.

---

# 7. Example

Suppose exam scores are approximately normally distributed:

$$
\mu=70
$$

$$
\sigma=10
$$

### Within 1 SD

$$
70\pm10
$$

So:

$$
60 \text{ to } 80
$$

Approximately **68%** of students fall in this range.

### Within 2 SD

$$
70\pm20
$$

So:

$$
50 \text{ to } 90
$$

Approximately **95%**.

### Within 3 SD

$$
70\pm30
$$

So:

$$
40 \text{ to } 100
$$

Approximately **99.7%**.

---

# 8. Standard Normal Distribution

A **standard normal distribution** has:

$$
\mu=0
$$

and:

$$
\sigma=1
$$

It is commonly represented as:

$$
Z\sim\mathcal{N}(0,1)
$$

To convert a value into a standard score, we use the **Z-score**:

$$
\boxed{
Z=\frac{x-\mu}{\sigma}
}
$$

Example:

$$
x=80,\quad\mu=70,\quad\sigma=10
$$

Then:

$$
Z=\frac{80-70}{10}
$$

$$
Z=1
$$

So the value is **1 standard deviation above the mean**.

---

# 9. Why Normal Distribution Matters in ML

Normal distribution is important for:

### Statistics

Many statistical techniques are based on normality assumptions under specific conditions.

### Standardization

A common transformation is:

$$
Z=\frac{x-\mu}{\sigma}
$$

which is the basis of standard scaling.

### Outlier Detection

Z-scores can help identify unusually extreme observations, although the method should be used appropriately.

### Statistical Modeling

Normal distributions appear in many statistical models and assumptions.

---

# 10. Python Example

Using NumPy:

```python
import numpy as np

data = np.random.normal(
    loc=70,
    scale=10,
    size=1000
)

print(np.mean(data))
print(np.std(data))
```

Approximately:

```text
Mean ≈ 70
Std ≈ 10
```

Because random sampling introduces variation, the values will not be exactly 70 and 10.

---

## Key Points

- Normal distribution is a **continuous, bell-shaped distribution**.
- It is symmetric around the mean.
- For a perfectly normal distribution:

$$
\text{Mean} = \text{Median} = \text{Mode}
$$

- Mean controls the **center**.
- Standard deviation controls the **spread**.
- Approximately:
  - **68%** → within ±1σ
  - **95%** → within ±2σ
  - **99.7%** → within ±3σ
- Standard normal distribution:

$$
\mu=0,\quad\sigma=1
$$

- Z-score:

$$
Z=\frac{x-\mu}{\sigma}
$$

- Normal distribution is important in **statistics, standardization, outlier analysis, and many ML/statistical methods**.