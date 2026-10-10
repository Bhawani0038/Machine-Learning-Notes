# Module 2 — Mathematics & Statistics for Machine Learning
## Topic 3: Range, Variance, and Standard Deviation

### What We Will Cover

- Range
- Variance
- Standard Deviation
- Population vs Sample Variance
- Intuition behind Variance and SD
- Simple examples
- Python implementation
- Why ML needs these concepts
- Key Points

---

# 1. Range

**Range tells us how far the smallest and largest values are apart.**

### Formula

$$
Range = Maximum - Minimum
$$

### Example

```text id="n0k6wq"
10, 20, 30, 40, 50
```

Maximum:

```text
50
```

Minimum:

```text
10
```

Therefore:

$$
Range = 50 - 10 = 40
$$

**Range = 40**

---

# 2. Limitation of Range

Range only considers **two values**:

- Minimum
- Maximum

It ignores everything in between.

Consider:

```text id="9p8t9v"
Dataset A:
10, 20, 30, 40, 50

Dataset B:
10, 10, 10, 50, 50
```

Both have:

```text
Range = 40
```

But their distributions are clearly different.

Therefore, we need a better measure of spread.

---

# 3. Variance

**Variance measures how far the values are spread from the mean.**

The basic idea is:

```text id="oh6aet"
Value
  ↓
Difference from Mean
  ↓
Square the Difference
  ↓
Average the Squared Differences
  ↓
Variance
```

---

# 4. Variance Example

Consider:

```text id="b36y6d"
10, 20, 30
```

### Step 1 — Calculate Mean

$$
Mean = \frac{10+20+30}{3}=20
$$

### Step 2 — Calculate Differences

| Value | Mean | Difference |
|---:|---:|---:|
| 10 | 20 | -10 |
| 20 | 20 | 0 |
| 30 | 20 | +10 |

### Step 3 — Square the Differences

| Value | Difference | Squared Difference |
|---:|---:|---:|
| 10 | -10 | 100 |
| 20 | 0 | 0 |
| 30 | 10 | 100 |

### Step 4 — Calculate Variance

For a population:

$$
Variance = \frac{100+0+100}{3}
$$

$$
Variance = 66.67
$$

So:

**Variance ≈ 66.67**

---

# 5. Why Do We Square the Differences?

Suppose we don't square them:

```text id="q2y8y0"
-10 + 0 + 10 = 0
```

The positive and negative differences cancel each other.

By squaring:

```text id="n9ehk6"
(-10)² = 100
0² = 0
10² = 100
```

All distances become positive.

That's why variance uses **squared differences**.

---

# 6. Variance Formula

### Population Variance

$$
\sigma^2 =
\frac{\sum (x_i-\mu)^2}{N}
$$

Where:

- `σ²` = Population variance
- `xᵢ` = Individual value
- `μ` = Population mean
- `N` = Number of values

---

# 7. Sample Variance

When calculating variance from a **sample**, we commonly divide by `n - 1`.

$$
s^2 =
\frac{\sum (x_i-\bar{x})^2}{n-1}
$$

The `n - 1` adjustment is called **Bessel's correction**.

For basic ML work, remember:

```text id="2udxov"
Population → divide by N

Sample → divide by n - 1
```

---

# 8. Standard Deviation

**Standard deviation is the square root of variance.**

$$
SD = \sqrt{Variance}
$$

For our example:

$$
Variance = 66.67
$$

Therefore:

$$
SD = \sqrt{66.67}
$$

Approximately:

$$
SD \approx 8.16
$$

---

# 9. Why Standard Deviation Is Easier to Understand

Suppose:

```text id="m4v8y6"
Height → measured in cm
```

Variance would be in:

```text
cm²
```

But standard deviation is in:

```text
cm
```

So standard deviation has the **same unit as the original data**.

This makes it easier to interpret.

---

# 10. Python Implementation

Using NumPy:

```python id="xw9m1m"
import numpy as np

data = np.array([10, 20, 30])

print("Mean:", np.mean(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))
```

Output:

```text id="gr8nd5"
Mean: 20.0
Variance: 66.66666666666667
Standard Deviation: 8.16496580927726
```

NumPy's default `np.var()` and `np.std()` use the **population-style divisor `N`**.

---

# 11. Sample Variance in NumPy

To calculate sample variance:

```python id="7k2w8x"
np.var(data, ddof=1)
```

Sample standard deviation:

```python id="0j4k9x"
np.std(data, ddof=1)
```

Here:

```text id="z2h5a4"
ddof=1
```

means the calculation uses `n - 1`.

---

# 12. Understanding Standard Deviation Intuitively

Consider two classes.

### Class A

```text id="8w0v7e"
48, 49, 50, 51, 52
```

Values are close together.

### Class B

```text id="t3x2kg"
20, 35, 50, 65, 80
```

Values are much more spread out.

Both have:

```text id="q3z2k6"
Mean = 50
```

But:

```text id="8b8o4n"
Class A → Low Standard Deviation
Class B → High Standard Deviation
```

So:

> **Higher standard deviation → more spread**

> **Lower standard deviation → less spread**

---

# 13. Why This Matters in Machine Learning

Many ML algorithms are affected by the **scale and spread of features**.

Suppose:

| Feature | Values |
|---|---:|
| Age | 20–60 |
| Salary | 20,000–2,00,000 |
| House Price | 30–150 lakh |

These features have very different scales.

Standardization uses:

$$
z = \frac{x-\mu}{\sigma}
$$

This transforms a value based on its distance from the mean measured in standard deviations.

We will study **standardization and Z-score** in detail later.

---

# Quick Comparison

| Measure | What it tells us |
|---|---|
| Range | Difference between maximum and minimum |
| Variance | Average squared distance from mean |
| Standard Deviation | Typical spread around the mean |

### Relationship

```text id="j4j2x9"
Variance
   ↓ √
Standard Deviation
```

---

## Key Points

- **Range = Maximum − Minimum**
- Range is simple but uses only minimum and maximum.
- **Variance measures spread using squared differences from the mean.**
- **Standard deviation = √Variance**
- Low SD → values are closer together.
- High SD → values are more spread out.
- Variance is measured in squared units.
- Standard deviation has the same unit as the original data.
- Population variance uses `N`.
- Sample variance commonly uses `n − 1`.
- Standard deviation and variance are important concepts for **scaling, preprocessing, statistics, and ML algorithms**.