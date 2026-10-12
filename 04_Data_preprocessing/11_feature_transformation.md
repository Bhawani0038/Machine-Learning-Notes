## Topic 11: Feature Transformation

**Objective\:** Learn how to transform numerical features to reduce skewness, handle highly uneven distributions, and improve the suitability of data for certain machine learning algorithms.

We will continue with our Customer Churn dataset, particularly `monthly_charges` and `support_calls`\.

## 1\. What is feature transformation?

Feature transformation changes the mathematical representation of a feature.

For example, monthly charges might contain many ordinary values and a few extremely large values. A logarithmic transformation can compress the large values and reduce right skew.

Consider these original values\:

| Original value | `log1p(value)` |
|--------|--------|
| 0 | 0.000 |
| 1 | 0.693 |
| 10 | 2.398 |
| 100 | 4.615 |
| 1,000 | 6.909 |
| 10,000 | 9.210 |

Notice that the differences between large values become smaller after transformation.

## 2\. Transformation vs. scaling

| Feature transformation | Feature scaling |
|--------|--------|
| Changes the mathematical representation or distribution | Changes the numerical scale |
| Example: logarithmic transformation | Example: StandardScaler |
| Can reduce skewness | Does not generally remove skewness |
| May change the relationship between values | Usually applies a linear rescaling |

These techniques can be used together when appropriate.

## 3\. Log transformation

A logarithmic transformation is useful for certain positive, right-skewed features.

### A. Understand `np.log()`

```python
import numpy as np

values = np.array([1, 10, 100, 1000])

print(np.log(values))
```

`np.log()` calculates the natural logarithm.

However, it cannot directly handle zero or negative values as ordinary real numbers.

### B. Use `np.log1p()`

For non-negative data that may contain zeros, use\:

```python
df["support_calls_log"] = np.log1p(
    df["support_calls"]
)

print(
    df[["support_calls", "support_calls_log"]].head()
)
```

`np.log1p(x)` computes\:

$$
x'=\ln(1+x)
$$

This makes the transformation defined for zero and compresses larger values.

**Important\:** `log1p()` is not a solution for negative values or missing values. Investigate invalid values and handle missing data appropriately first.

### C. Transform monthly charges

```python
df["monthly_charges_log"] = np.log1p(
    df["monthly_charges"]
)
```

Use this transformation only if the distribution and business context justify it. For a feature with negative values, a different strategy may be needed.

## 4\. Square-root transformation

The square-root transformation is another option for non-negative, right-skewed features.

$$
x'=\sqrt{x}
$$

Example\:

```python
df["support_calls_sqrt"] = np.sqrt(
    df["support_calls"]
)
```

Compare the outputs\:

| Original value | Square root |
|--------|--------|
| 0 | 0.00 |
| 1 | 1.00 |
| 4 | 2.00 |
| 9 | 3.00 |
| 16 | 4.00 |

Square-root transformation compresses large values, generally less aggressively than a logarithmic transformation.

## 5\. Visualize before and after transformation

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(
    df["monthly_charges"].dropna(),
    ax=axes[0],
    kde=True
)
axes[0].set_title("Original Monthly Charges")

sns.histplot(
    np.log1p(df["monthly_charges"].dropna()),
    ax=axes[1],
    kde=True
)
axes[1].set_title("Log-Transformed Monthly Charges")

plt.tight_layout()
plt.show()
```

Look for changes in skewness and the spread of observations. A transformed distribution does not have to become perfectly normal to be useful.

## 6\. Power transformations in scikit-learn

Scikit-learn provides transformations that estimate parameters from data.

### A. `PowerTransformer`

```python
from sklearn.preprocessing import PowerTransformer

power_transformer = PowerTransformer(
    method="yeo-johnson",
    standardize=False
)

transformed_values = power_transformer.fit_transform(
    df[["monthly_charges", "support_calls"]]
)
```

The Yeo-Johnson method supports both positive and negative numerical values. It does not automatically make every feature normally distributed.

Another option, `method="box-cox"`, requires strictly positive values.

**Machine learning rule\:** Fit data-dependent transformation parameters on the training set only, then apply the fitted transformation to validation and test data. We will implement this safely using a preprocessing pipeline.

## 7\. When should you transform a feature?

| Situation | Possible approach |
|--------|--------|
| Strong right skew | Log or power transformation |
| Non-negative counts with moderate skew | Square root or power transformation |
| Negative and positive values with skew | Yeo-Johnson |
| Approximately symmetric data | Transformation may not be necessary |
| Tree-based models | Transformation is often unnecessary, but evaluate the actual problem |

Transformation is not mandatory for every numerical column. Always compare model performance and retain interpretability where possible.