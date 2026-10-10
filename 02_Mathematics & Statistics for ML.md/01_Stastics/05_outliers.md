# Module 2 — Mathematics & Statistics for Machine Learning
## Topic 5: Outliers

### What We Will Cover

- What is an Outlier?
- Examples
- Causes of Outliers
- Outlier vs Normal Variation
- IQR Method
- Z-Score Method
- Effect on ML Models
- How to Handle Outliers
- When Not to Remove Outliers
- Key Points

---

# 1. What is an Outlier?

An **outlier is an observation that is unusually far from the majority of the data**.

Example:

```text
20, 22, 24, 25, 27, 26, 100
```

Most values are around `20–27`.

```text
100
```

is unusually large and may be an **outlier**.

---

# 2. Real-Life Example — Salary

Suppose salaries are:

```text
₹30,000
₹35,000
₹38,000
₹42,000
₹45,000
₹48,000
₹5,00,000
```

Most salaries are between ₹30,000 and ₹48,000.

₹5,00,000 is much larger.

It may be an outlier.

---

# 3. Why Do Outliers Occur?

An outlier can occur because of:

### 1. Data Entry Error

Actual age:

```text
35
```

Entered as:

```text
350
```

### 2. Measurement Error

A faulty sensor may record:

```text
Temperature = 250°C
```

when the actual temperature is 25°C.

### 3. Rare Event

Sometimes the value is completely correct but unusual.

Example:

```text
Most transactions = ₹500–₹5,000
One transaction = ₹5,00,000
```

This could be a genuine high-value purchase.

### 4. Fraud

In banking:

```text
Normal transaction = ₹2,000
Unusual transaction = ₹10,00,000
```

The unusual value may actually be the most important observation.

---

# 4. Important: Outlier ≠ Error

An outlier is **not automatically wrong**.

For example:

```text
House prices:
₹40 lakh
₹45 lakh
₹50 lakh
₹55 lakh
₹5 crore
```

The ₹5 crore property may be a genuine luxury property.

Therefore:

> **Don't automatically delete every outlier.**

First understand why it exists.

---

# 5. Detecting Outliers Using IQR

One common method uses the **Interquartile Range**.

### Step 1 — Calculate Q1 and Q3

```text id="h6t9jm"
Q1 = 25th percentile
Q3 = 75th percentile
```

### Step 2 — Calculate IQR

$$
IQR = Q3-Q1
$$

### Step 3 — Calculate boundaries

$$
Lower = Q1 - 1.5(IQR)
$$

$$
Upper = Q3 + 1.5(IQR)
$$

Values outside these boundaries are commonly flagged as **potential outliers**.

---

# 6. Example

Suppose:

```text id="f2krq5"
Q1 = 20
Q3 = 40
```

Then:

$$
IQR = 40-20 = 20
$$

Lower boundary:

$$
20-(1.5\times20)=-10
$$

Upper boundary:

$$
40+(1.5\times20)=70
$$

Therefore:

```text id="e9upqf"
< -10 → Potential outlier
> 70  → Potential outlier
```

---

# 7. Detecting Outliers with Python

```python id="mlq6v7"
import numpy as np

data = np.array([10, 20, 25, 30, 35, 40, 100])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[(data < lower) | (data > upper)]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Outliers:", outliers)
```

---

# 8. Z-Score Method

Another common approach is the **Z-score**.

The Z-score tells us how many standard deviations a value is away from the mean.

### Formula

$$
Z = \frac{x-\mu}{\sigma}
$$

Where:

- `x` = observation
- `μ` = mean
- `σ` = standard deviation

For example:

```text id="l6c0ty"
Z = 0
```

means the value is at the mean.

```text id="l9ph6a"
Z = +2
```

means approximately 2 standard deviations above the mean.

```text id="7l4ps7"
Z = -2
```

means approximately 2 standard deviations below the mean.

A commonly used rule of thumb flags observations with:

```text id="r1m0f4"
|Z| > 3
```

as potential extreme outliers, particularly when the data is approximately normally distributed.

---

# 9. IQR vs Z-Score

| IQR | Z-Score |
|---|---|
| Based on Q1 and Q3 | Based on mean and SD |
| More robust to outliers | More affected by outliers |
| Useful for skewed data | Works well with approximately normal data |
| Uses 1.5 × IQR rule | Often uses `|Z| > 3` |
| Common in EDA | Common in statistical analysis |

---

# 10. Effect of Outliers on Machine Learning

Outliers can affect some ML algorithms significantly.

### Linear Regression

An extreme observation can pull the regression line toward itself.

```text id="5q3x2w"
Normal Data
    ↘
     ↘
      ↘
       ● Outlier
```

The model may try to accommodate the extreme observation.

### KNN

KNN uses distances.

An extreme value can affect distances between observations.

### K-Means

K-Means uses distances and centroids.

An extreme point can pull the centroid away from the main group.

---

# 11. Some Models Are More Robust

Tree-based algorithms such as:

- Decision Tree
- Random Forest

are generally less sensitive to feature scaling and certain extreme numeric values than distance-based models or linear models.

However, **outliers can still represent data-quality problems**, so they should not simply be ignored.

---

# 12. What Can We Do With Outliers?

There is no single correct solution.

### Option 1 — Keep Them

Use this when the value is:

- Correct
- Meaningful
- Representative of a real situation

---

### Option 2 — Remove Them

Consider removing an observation when it is clearly:

- Data-entry error
- Measurement error
- Corrupted data

Example:

```text
Age = 350
```

---

### Option 3 — Correct the Value

If the correct value can be identified:

```text
350 → 35
```

---

### Option 4 — Cap/Winsorize

Very extreme values can sometimes be replaced with a chosen upper/lower threshold.

For example:

```text
Very high values
       ↓
99th percentile
       ↓
Cap values
```

This should be done carefully because it changes the data.

---

### Option 5 — Transform the Data

For strongly skewed variables, transformations such as a logarithm can reduce the influence of very large values.

Example:

```text
income
  ↓
log(income)
```

We will study transformations later in **Data Preprocessing**.

---

# 13. Outliers in EDA

Outlier detection is usually part of **Exploratory Data Analysis (EDA)**.

Typical workflow:

```text id="0ce4h8"
Load Data
    ↓
Understand Data
    ↓
Check Distribution
    ↓
Detect Outliers
    ↓
Investigate Cause
    ↓
Decide What to Do
```

The important step is:

> **Investigate before removing.**

---

## Key Points

- **Outlier = unusually distant observation.**
- An outlier is **not automatically an error**.
- Common causes: errors, rare events, fraud, genuine extreme cases.
- **IQR method:** values outside `Q1 − 1.5×IQR` and `Q3 + 1.5×IQR` are commonly flagged.
- **Z-score:** measures distance from the mean in standard-deviation units.
- `|Z| > 3` is a common rule of thumb for extreme values.
- Outliers can affect **mean, variance, regression, KNN, and K-Means**.
- Don't blindly delete outliers.
- First determine whether the observation is **wrong, rare, or genuinely important**.