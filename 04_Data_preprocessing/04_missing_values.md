## Topic 4: Missing Values

**Objective\:** Learn how to identify missing values, measure their impact, choose an appropriate handling strategy, and apply imputation using pandas and Scikit-learn.

We will continue using our Customer Churn dataset.

## 1\. What are missing values?

Missing values occur when information is unavailable for a particular observation.

For example\:

| customer\_id | age | monthly\_charges | gender |
|--------|--------|--------|--------|
| C001 | 25 | 500 | Male |
| C002 | 35 | 1200 | Female |
| C003 | 45 | 800 | Male |
| C004 | NaN | 1500 | Female |
| C005 | 29 | NaN | NaN |

In pandas, missing values are commonly represented by `NaN`, `None`, or `pd.NA`\.

## 2\. Detect missing values

### A. Count missing values in each column

```python
print(df.isnull().sum())
```

- `isnull()` identifies missing cells.
- `sum()` counts missing cells in each column.

### B. Calculate missing-value percentages

```python
missing_percentage = (
    df.isnull().mean() * 100
).round(2)

print(missing_percentage)
```

The percentage helps us compare missingness across columns, even when datasets have different numbers of rows.

### C. Display only rows containing missing values

```python
missing_rows = df[df.isnull().any(axis=1)]

print(missing_rows)
```

Here, `axis=1` means that we check across the columns of each row.

## 3\. Choose a missing-value handling strategy

| Feature type | Common strategy | Example |
|--------|--------|--------|
| Numerical, roughly symmetric | Mean imputation | Age |
| Numerical, skewed or containing outliers | Median imputation | Monthly charges |
| Categorical | Most frequent value | Gender |
| Missingness is meaningful | Missing category or indicator | Unknown payment method |
| Too many missing values | Investigate whether to exclude the feature | A mostly empty column |

These are starting points, not universal rules. The appropriate strategy depends on the data distribution, business meaning, and reason values are missing.

**Important\:** Never assume that missing values mean zero. A missing monthly charge does not necessarily mean that the customer paid nothing.

## 4\. Method 1 - Handle missing values using pandas

Pandas provides `fillna()` for replacing missing values.

### A. Fill numerical values with the median

```python
df["age"] = df["age"].fillna(
    df["age"].median()
)

df["monthly_charges"] = df["monthly_charges"].fillna(
    df["monthly_charges"].median()
)
```

The median is often preferable when numerical data contains extreme values because it is less sensitive to outliers than the mean.

### B. Fill categorical values with the mode

```python
df["gender"] = df["gender"].fillna(
    df["gender"].mode()[0]
)
```

`mode()` returns the most frequent value or values. `[0]` selects the first mode.

### C. Verify the result

```python
print(df.isnull().sum())
```

**Limitation\:** If we calculate medians and modes using the entire dataset before splitting into training and test sets, information from the test set influences preprocessing. This can cause data leakage.

For a machine learning workflow, we will later use Scikit-learn imputers fitted only on training data.

## 5\. Method 2 - Use Scikit-learn imputers

Scikit-learn provides reusable preprocessing components.

Install the required libraries if needed\:

```bash
pip install pandas numpy scikit-learn
```

### A. Numerical imputation

```python
from sklearn.impute import SimpleImputer

numerical_columns = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

numerical_imputer = SimpleImputer(
    strategy="median"
)

df[numerical_columns] = numerical_imputer.fit_transform(
    df[numerical_columns]
)
```

The `median` strategy calculates a median for each numerical column.

Other supported strategies include\:

- `"mean"` - replace missing values with the mean.
- `"median"` - replace missing values with the median.
- `"most_frequent"` - use the most frequent value.
- `"constant"` - use a specified value, such as `"Unknown"`\.

### B. Categorical imputation

```python
categorical_columns = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

df[categorical_columns] = categorical_imputer.fit_transform(
    df[categorical_columns]
)
```

This example demonstrates imputation on the existing dataset. In a production ML workflow, we should split the data first and fit imputers on training data only; later, we will integrate this process into a `Pipeline` and `ColumnTransformer`\.

## 6\. Mean vs. median imputation

Consider the following ages\:

```python
ages = [25, 28, 30, 32, 35, 150]

print("Mean:", np.mean(ages))
print("Median:", np.median(ages))
```

Expected output\:

```text
Mean: 50.0
Median: 31.0
```

The extreme age of 150 pulls the mean upward, while the median remains closer to the central group of values.

However, imputation should not be used to hide invalid records. An age of 150 needs investigation separately.