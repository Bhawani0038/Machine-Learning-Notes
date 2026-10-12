## Topic 2: Numerical Features

**Objective\:** Learn how to inspect numerical columns, understand their distributions, identify unusual values, and decide which preprocessing techniques are appropriate.

We will continue with the same Customer Churn dataset.

### 1\. Identify numerical features

Numerical features contain measurable quantities or counts.

```python
numerical_columns = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

print(df[numerical_columns].head())
```

These columns have different meanings\:

| Feature | Type | Example |
|--------|--------|--------|
| `age` | Continuous-like | 35 |
| `monthly_charges` | Continuous | 1200 |
| `tenure_months` | Discrete | 24 |
| `support_calls` | Discrete count | 3 |

Although `tenure_months` contains numbers, we previously converted it to strings. We should correct its type before numerical analysis.

```python
df["tenure_months"] = pd.to_numeric(
    df["tenure_months"],
    errors="coerce"
)
```

`errors="coerce"` converts values that cannot be parsed into numbers to `NaN`, which we can investigate later.

## 2\. Understand numerical distributions

### A. Descriptive statistics

```python
print(df[numerical_columns].describe().T)
```

Important statistics\:

| Statistic | Meaning |
|--------|--------|
| `count` | Number of non-missing observations |
| `mean` | Arithmetic average |
| `std` | Standard deviation |
| `min` | Minimum value |
| `25%` | First quartile (Q1) |
| `50%` | Median (Q2) |
| `75%` | Third quartile (Q3) |
| `max` | Maximum value |

For example, if most customers pay ₹500-₹2,200 monthly but one customer has a charge of ₹50,000, the maximum and mean may be unusually high. That observation deserves investigation.

### B. Check the range of each feature

```python
for column in numerical_columns:
    print(
        column,
        "Minimum:", df[column].min(),
        "Maximum:", df[column].max()
    )
```

Ranges help identify values that may be invalid or unusual. However, an unusual value is not automatically incorrect.

### C. Count missing numerical values

```python
print(df[numerical_columns].isna().sum())
```

Missing values can affect calculations, visualizations, and model training. We will learn how to handle them in Topic 4.

## 3\. Visualize numerical features

Install the required libraries if needed: `pip install pandas numpy matplotlib seaborn`\.

### A. Histograms - understand distributions

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=df, x="monthly_charges", bins=10)
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges (₹)")
plt.ylabel("Number of Customers")
plt.show()
```

A histogram helps you identify\:

- Where most observations are concentrated.
- Whether the distribution is symmetric or skewed.
- Whether extreme values stretch the distribution.

The ₹50,000 charge may compress the rest of the histogram. After investigating its validity, we can visualize the ordinary range separately if needed.

### B. Box plots - inspect potential outliers

```python
sns.boxplot(data=df, x="monthly_charges")
plt.title("Monthly Charges - Box Plot")
plt.show()
```

A box plot displays the median, quartiles, and potential outliers.

The conventional IQR rule identifies potential outliers using\:

$$
IQR = Q_3-Q_1
$$

$$
\text{Lower Bound}=Q_1-1.5(IQR)
$$

$$
\text{Upper Bound}=Q_3+1.5(IQR)
$$

Values outside these bounds are flagged for investigation, not automatically deleted.

## 4\. Detect potential outliers programmatically

```python
column = "monthly_charges"

q1 = df[column].quantile(0.25)
q3 = df[column].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df[column] < lower_bound) |
    (df[column] > upper_bound)
]

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print(outliers[["customer_id", column]])
```

This code detects unusual monthly charges using the IQR method. We can later apply the same logic to age and support calls.