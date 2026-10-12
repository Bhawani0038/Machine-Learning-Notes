## Topic 7: Outliers

**Objective\:** Learn how to identify extreme numerical observations, understand their impact on machine learning models, and decide whether to retain, remove, cap, or transform them.

We will use the same Customer Churn dataset, which contains deliberately unusual values such as age `150`, monthly charges `50000`, and support calls `20`\.

## 1\. What is an outlier?

An outlier is an observation that differs substantially from most other observations in a dataset.

Example\:

```python
import pandas as pd

df_outlier = pd.DataFrame({
    "monthly_charges": [
        500, 600, 650, 700, 750,
        800, 850, 900, 1000, 50000
    ]
})

print(df_outlier.describe())
```

Most charges are between ₹500 and ₹1,000, but one observation is ₹50,000.

Possible explanations include\:

- A data-entry error.
- A currency or unit mismatch.
- A legitimate premium customer.
- An unusual but valid business event.

**Important\:** An outlier is not necessarily an error. Investigate the observation before changing it.

## 2\. Method 1 - Detect outliers using the IQR method

The Interquartile Range (IQR) method identifies values outside a range calculated from the first and third quartiles.

$$
IQR=Q_3-Q_1
$$

$$
\text{Lower Bound}=Q_1-1.5(IQR)
$$

$$
\text{Upper Bound}=Q_3+1.5(IQR)
$$

Values outside these bounds are flagged as potential outliers.

### Python implementation

```python
column = "monthly_charges"

q1 = df_outlier[column].quantile(0.25)
q3 = df_outlier[column].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df_outlier[
    (df_outlier[column] < lower_bound) |
    (df_outlier[column] > upper_bound)
]

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("\nPotential outliers:")
print(outliers)
```

### How it works

1. Calculate Q1 and Q3.
2. Calculate the IQR.
3. Determine the lower and upper bounds.
4. Identify observations outside those bounds.

The IQR method is useful when the data is skewed or contains extreme observations because quartiles are relatively resistant to extreme values.

## 3\. Method 2 - Detect outliers using Z-scores

A Z-score measures how far an observation is from the mean in standard deviation units.

$$
z=\frac{x-\mu}{\sigma}
$$

Where\:

- $x$ \= observation
- $\mu$ \= mean
- $\sigma$ \= standard deviation

A common rule flags observations with $|z|>3$\.

```python
import numpy as np
from scipy.stats import zscore

values = df_outlier["monthly_charges"]

z_scores = zscore(values, ddof=0)

outliers_zscore = df_outlier[
    np.abs(z_scores) > 3
]

print(outliers_zscore)
```

Install SciPy if necessary\:

```bash
pip install scipy
```

**Limitation\:** Z-scores can be misleading for highly skewed data because the mean and standard deviation are themselves influenced by extreme observations. The threshold of 3 is a heuristic, not a universal rule.

## 4\. Visualize outliers using a box plot

```python

import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(data=df_outlier, x="monthly_charges")

plt.title("Outlier Detection: Monthly Charges")
plt.xlabel("Monthly Charges (₹)")
plt.show()
```

A box plot displays\:

- Median: the central line.
- Box: the middle 50% of observations.
- Whiskers: the range covered under the plotting rule.
- Individual points beyond the whiskers: potential outliers.

## 5\. What should we do with outliers?

| Strategy | When to use it | Example |
|--------|--------|--------|
| Retain | The observation is valid and informative | A genuine high-value customer |
| Remove | The observation is confirmed to be erroneous and cannot be corrected | An impossible measurement |
| Capping | Extreme values are valid but their influence needs limiting | A highly skewed financial feature |
| Transformation | The distribution is strongly right-skewed | Monthly charges or transaction amounts |
| Robust scaling | Features contain outliers and the model benefits from scaling | `RobustScaler` |

### A. Remove confirmed erroneous observations

```python
df_clean = df_outlier[
    df_outlier["monthly_charges"] <= 10000
].copy()
```

This example removes values above ₹10,000 only as an illustration. In real projects, the threshold must be justified by business rules or validated data.

### B. Cap values using the IQR bounds

```python
df_capped = df_outlier.copy()

df_capped["monthly_charges"] = (
    df_capped["monthly_charges"]
    .clip(lower=lower_bound, upper=upper_bound)
)
```

Capping replaces values outside the specified bounds with the corresponding boundary values. It preserves the number of records but changes their values.

### C. Transform right-skewed data

```python
df_transformed = df_outlier.copy()

df_transformed["log_monthly_charges"] = np.log1p(
    df_transformed["monthly_charges"]
)
```

`np.log1p(x)` calculates $\log(1+x)$, which can reduce right skew for non-negative data. It does not remove outliers, and the transformed values need to be interpreted on the log scale.

## 6\. Important machine learning rule

For supervised machine learning, **split your data before learning outlier thresholds**\. Calculate IQR bounds or other data-dependent thresholds from the training set and apply those same thresholds to the test set.

Also, investigate data quality issues before treating statistical outliers. An impossible age and a genuine high monthly charge should not necessarily receive the same treatment.