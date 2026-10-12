# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 17: Z-Score & Standard Normal Distribution

### What We Will Cover
- What is Z-Score?
- Why Z-Score is Needed
- Formula
- Positive & Negative Z-Scores
- Interpretation
- Simple Examples
- Z-Score and Outliers
- Standardization in ML
- Python Implementation
- Key Points

---

## 1. What is Z-Score?

A **Z-score tells us how many standard deviations a value is away from the mean.**

Formula:

$$
\boxed{Z=\frac{x-\mu}{\sigma}}
$$

Where:

- $x$ = individual value
- $\mu$ = mean
- $\sigma$ = standard deviation

---

## 2. Simple Example

Suppose exam scores have:

$$
\mu=70
$$

$$
\sigma=10
$$

A student scores:

$$
x=80
$$

Then:

$$
Z=\frac{80-70}{10}
$$

$$
Z=1
$$

### Interpretation

The student's score is:

> **1 standard deviation above the mean.**

---

# 3. Negative Z-Score

Suppose another student scores:

$$
x=50
$$

Then:

$$
Z=\frac{50-70}{10}
$$

$$
Z=-2
$$

This means:

> The score is **2 standard deviations below the mean**.

---

# 4. Understanding Z-Score

| Z-Score | Meaning |
|---:|---|
| `0` | Exactly at mean |
| `+1` | 1 SD above mean |
| `+2` | 2 SD above mean |
| `+3` | 3 SD above mean |
| `-1` | 1 SD below mean |
| `-2` | 2 SD below mean |
| `-3` | 3 SD below mean |

So:

```text id="v2mrxh"
       Below Mean       Above Mean

 -3   -2   -1    0    +1   +2   +3
 ─────┴────┴────┴────┴────┴────┴────
                  ↑
                Mean
```

---

# 5. Why Do We Need Z-Scores?

Suppose two students have:

### Student A

```text
Marks = 80
```

### Student B

```text
Marks = 80
```

The raw scores are the same.

But suppose:

```text id="1n1f1f"
Class A:
Mean = 70
SD = 10

Class B:
Mean = 50
SD = 5
```

For Student A:

$$
Z_A=\frac{80-70}{10}=1
$$

For Student B:

$$
Z_B=\frac{80-50}{5}=6
$$

Both scored 80, but relative to their classes, Student B is much farther above the mean.

This is the power of the Z-score.

---

# 6. Z-Score Standardization

Z-score transformation converts data into a standardized scale.

Formula:

$$
Z=\frac{x-\mu}{\sigma}
$$

After standardization, the transformed data has approximately:

$$
\text{Mean}=0
$$

$$
\text{Standard Deviation}=1
$$

Example:

```text id="8q7lpg"
Original Data
10, 20, 30, 40, 50

        ↓

Z-Score Standardization

-1.41, -0.71, 0, 0.71, 1.41
```

---

# 7. Z-Score in Machine Learning

Suppose an ML model has:

```text id="j5q3j9"
Age             → 18 to 70
Annual Income   → 20,000 to 2,000,000
```

Income has a much larger numerical scale.

Standardization transforms each feature using:

$$
Z=\frac{x-\mu}{\sigma}
$$

This can be particularly useful for algorithms that are sensitive to feature scale, such as:

- KNN
- K-Means
- SVM
- Logistic Regression
- Linear Regression with certain optimization methods
- PCA
- Neural Networks

Tree-based models generally do not require standardization for their split decisions.

---

# 8. Z-Score and Outliers

For data that is approximately normally distributed, very large absolute Z-scores can indicate unusual observations.

A common rule of thumb is:

$$
|Z|>3
$$

may indicate a potential extreme outlier.

Example:

$$
Z=4.2
$$

This observation is unusually far from the mean.

### Important

A large Z-score does **not automatically mean the value is wrong**.

It may be a genuine extreme observation.

---

# 9. Python — Manual Calculation

```python id="xq4f3j"
x = 80
mean = 70
std = 10

z = (x - mean) / std

print(z)
```

Output:

```text id="c2xqf8"
1.0
```

---

# 10. Python — NumPy

```python id="8ikp3k"
import numpy as np

data = np.array([50, 60, 70, 80, 90])

z_scores = (data - np.mean(data)) / np.std(data)

print(z_scores)
```

The resulting values show how far each observation is from the mean in standard-deviation units.

---

# 11. Scikit-learn StandardScaler

In ML, we commonly use:

```python id="f8e8s6"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Conceptually, `StandardScaler` applies:

$$
z=\frac{x-\mu}{\sigma}
$$

to each feature.

### Important ML Rule

When working with train/test data:

```text id="1b5b5k"
Training Data
     ↓
fit scaler
     ↓
transform training data

Test Data
     ↓
transform using the SAME fitted scaler
```

Do **not** calculate a separate mean and standard deviation from the test set.

Otherwise, you can introduce **data leakage**.

---

## Key Points

- Z-score tells us how many standard deviations a value is from the mean.
- Formula:

$$
\boxed{Z=\frac{x-\mu}{\sigma}}
$$

- Positive Z-score → above mean.
- Negative Z-score → below mean.
- `Z = 0` → exactly at mean.
- Standard normal distribution has:

$$
\mu=0,\quad\sigma=1
$$

- Z-score is useful for **standardization and identifying unusual observations**.
- `|Z| > 3` is a common outlier rule of thumb for approximately normal data.
- In ML, **fit the scaler on training data only**, then transform both training and test data using that fitted scaler.