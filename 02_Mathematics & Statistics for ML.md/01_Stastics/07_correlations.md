# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 7: Correlation

### What We Will Cover
- What is Correlation?
- Positive, Negative & Zero Correlation
- Pearson Correlation
- Correlation Coefficient
- Range: `-1 to +1`
- Correlation Strength
- Correlation vs Covariance
- Correlation ≠ Causation
- Outliers & Correlation
- Linear vs Non-Linear Relationship
- Python Implementation
- ML Examples

---

## 1. What is Correlation?

**Correlation measures the strength and direction of the relationship between two variables.**

Example:

| Study Hours | Marks |
|---:|---:|
| 1 | 35 |
| 2 | 45 |
| 3 | 55 |
| 4 | 65 |
| 5 | 75 |

As **study hours increase, marks also increase**.

So, they have a **positive correlation**.

---

## 2. Types of Correlation

### Positive Correlation

Both variables tend to move in the same direction.

```text
X ↑  →  Y ↑
X ↓  →  Y ↓
```

Example:

- Study Hours ↑ → Marks ↑
- House Area ↑ → Price ↑

Correlation:

```text
0 < r ≤ 1
```

---

### Negative Correlation

Variables tend to move in opposite directions.

```text
X ↑  →  Y ↓
X ↓  →  Y ↑
```

Example:

- Distance from city center ↑ → House Price may ↓
- Product Price ↑ → Demand may ↓

Correlation:

```text
-1 ≤ r < 0
```

---

### Zero Correlation

There is little or no **linear relationship** between the variables.

```text
r ≈ 0
```

Example:

```text
Shoe Size ↔ Exam Marks
```

There is generally no meaningful linear relationship.

---

# 3. Correlation Coefficient

The correlation coefficient is usually represented by:

```text
r
```

For Pearson correlation:

```text
-1 ≤ r ≤ +1
```

| `r` | Meaning |
|---:|---|
| `+1` | Perfect positive correlation |
| `+0.8` | Strong positive |
| `+0.5` | Moderate positive |
| `+0.2` | Weak positive |
| `0` | No linear correlation |
| `-0.2` | Weak negative |
| `-0.5` | Moderate negative |
| `-0.8` | Strong negative |
| `-1` | Perfect negative |

> These strength ranges are practical rules of thumb, not strict universal boundaries.

---

# 4. Pearson Correlation Formula

Pearson correlation is:

\[
r =
\frac{\text{Cov}(X,Y)}
{\sigma_X\sigma_Y}
\]

Where:

- `Cov(X,Y)` = covariance between X and Y
- `σX` = standard deviation of X
- `σY` = standard deviation of Y

### Important Idea

Correlation is essentially **standardized covariance**.

That's why correlation always stays between:

```text
-1 and +1
```

---

# 5. Correlation vs Covariance

| Covariance | Correlation |
|---|---|
| Measures joint movement | Measures strength and direction |
| Depends on units | Unit-free |
| Range is not fixed | Always `-1 to +1` |
| Harder to compare | Easier to compare |
| Used in mathematical calculations | Very useful in EDA and ML |

Example:

```text
sqft_area ↔ price
```

Covariance may be:

```text
250000
```

while correlation could be:

```text
0.87
```

The correlation value is much easier to interpret.

---

# 6. Python — NumPy

```python
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

correlation = np.corrcoef(x, y)[0, 1]

print(correlation)
```

Output:

```text
0.9999999999999999
```

Approximately:

```text
1
```

So X and Y have a **perfect positive linear relationship**.

---

# 7. Python — Pandas

```python
import pandas as pd

df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5],
    "marks": [35, 45, 55, 65, 75]
})

print(df["study_hours"].corr(df["marks"]))
```

Output:

```text
1.0
```

---

## 8. Correlation Matrix

For multiple variables:

```python
print(df.corr(numeric_only=True))
```

Example:

| | sqft_area | rooms | price |
|---|---:|---:|---:|
| sqft_area | 1.00 | 0.72 | 0.87 |
| rooms | 0.72 | 1.00 | 0.76 |
| price | 0.87 | 0.76 | 1.00 |

This tells us:

```text
sqft_area ↔ price = 0.87
rooms ↔ price     = 0.76
```

Both have a strong positive relationship with price.

---

# 9. Correlation ≠ Causation

This is **very important in Data Science**.

If:

```text
X and Y are highly correlated
```

it does **not automatically mean**:

```text
X causes Y
```

Example:

```text
Ice Cream Sales ↑
     +
Swimming Pool Accidents ↑
```

These may be correlated because:

```text
Hot Weather
     ↓
 ┌───┴────┐
 ↓        ↓
Ice Cream  Swimming
Sales      Activity
```

A third variable can influence both.

---

# 10. Outliers Can Affect Correlation

Example:

```text
Study Hours → Marks
```

Most students may show a strong positive relationship.

But one unusual data point can significantly change the correlation.

Therefore:

**Always inspect outliers before interpreting correlation.**

---

# 11. Correlation Detects Linear Relationships

A correlation near `0` does **not always mean there is no relationship**.

There could be a strong **non-linear relationship**.

For example:

\[
Y = X^2
\]

may have a weak Pearson correlation around certain symmetric ranges even though X and Y have a clear mathematical relationship.

So:

```text
Correlation ≈ 0
```

means:

> There is little/no **linear** relationship.

It does **not** necessarily mean:

> There is absolutely no relationship.

---

# 12. Why Correlation is Important in ML

Correlation is useful during **EDA and feature analysis**.

Example:

```text
sqft_area ───────→ price
rooms ───────────→ price
parking ─────────→ price
distance ────────→ price
```

We can examine correlations to understand which numerical features have stronger linear relationships with the target.

It can also help identify **highly correlated features with each other**, which may indicate multicollinearity in models such as linear regression.

---

## Key Points

- **Correlation measures relationship between two variables.**
- Pearson correlation ranges from **`-1 to +1`**.
- Positive → variables tend to move together.
- Negative → variables tend to move oppositely.
- `+1` → perfect positive linear relationship.
- `-1` → perfect negative linear relationship.
- `0` → no linear relationship.
- Correlation is **unit-free**.
- Correlation is standardized covariance.
- **Correlation does not imply causation.**
- Outliers can strongly affect correlation.
- Pearson correlation mainly measures **linear relationships**.
- Correlation is widely used during **EDA and feature analysis**.