## Topic 3: Categorical Features

**Objective\:** Learn how to identify categorical features, inspect their categories, find inconsistent labels, analyze category frequencies, and prepare them for encoding.

We will continue using the same Customer Churn dataset.

## 1\. What are categorical features?

Categorical features represent groups, labels, or classes rather than measurable quantities.

Examples from our dataset\:

| Feature | Possible categories | Type |
|--------|--------|--------|
| `gender` | Male, Female | Nominal |
| `contract_type` | Monthly, Yearly, Two Year | Ordinal-like |
| `internet_service` | Fiber, DSL, None | Nominal |
| `payment_method` | UPI, Card, Bank Transfer | Nominal |
| `churn` | Yes, No | Binary target |

**Nominal\:** Categories have no natural order.

**Ordinal\:** Categories have a meaningful order. For example, Low, Medium, High. Contract duration has a measurable length, but its category labels should not automatically be treated as a simple ordinal scale.

## 2\. Select categorical columns

```python
categorical_columns = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

print(df[categorical_columns].head())
```

We exclude `customer_id` because it is an identifier, and `churn` because it is our target variable.

## 3\. Inspect unique categories

Use `.unique()` to see the distinct values in each column.

```python
for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].unique())
```

For example, the `gender` column may contain\:

```text
['Male' 'female' 'Female' None]
```

Here, `female` and `Female` represent the same category but have different capitalization.

This inconsistency can cause encoding to treat them as separate categories.

### Count distinct categories

```python
for column in categorical_columns:
    print(column, ":", df[column].nunique())
```

By default, `nunique()` excludes missing values.

## 4\. Analyze category frequencies

Use `value_counts()` to understand how frequently each category occurs.

```python
for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts(dropna=False))
```

The `dropna=False` argument includes missing values in the frequency counts.

For example, this helps answer questions such as\:

- How many customers use fiber internet?
- How many customers pay through UPI?
- Are most customers on monthly contracts?
- Does a category occur only once?

### Visualize categorical frequencies

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=df, x="contract_type")
plt.title("Customer Contract Types")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.show()
```

A count plot is useful for understanding category distributions and identifying rare categories.

## 5\. Detect inconsistent category labels

Let's inspect the gender values.

```python
print(df["gender"].value_counts(dropna=False))
```

We can normalize capitalization and remove accidental leading or trailing spaces.

```python
df["gender"] = (
    df["gender"]
    .str.strip()
    .str.capitalize()
)

print(df["gender"].value_counts(dropna=False))
```

This converts `female` to `Female` and `male` to `Male`\.

**Important\:** Standardize labels only when the variations are genuinely equivalent. Do not combine distinct business categories merely because their names look similar.

## 6\. Identify high-cardinality features

Cardinality means the number of distinct categories in a feature.

```python
for column in categorical_columns:
    print(
        column,
        "Unique categories:",
        df[column].nunique()
    )
```

A column with hundreds or thousands of distinct categories may be considered high-cardinality.

Examples include\:

- City names
- Product IDs
- Customer IDs
- Postal codes

High-cardinality features require special consideration because one-hot encoding may generate too many columns. Also, identifiers such as `customer_id` usually should not be used as predictive features without a clear justification.

## 7\. Prepare categorical features for machine learning

Most scikit-learn estimators require numerical input features. We will later learn two important encoding techniques\:

- **One-Hot Encoding\:** Creates separate indicator columns for categories such as `payment_method`\.
- **Label Encoding\:** Assigns numerical labels to classes, commonly for target labels such as `churn`\.

Do not manually assign arbitrary numbers to nominal input categories and assume those numbers are meaningful. For example, encoding UPI = 1, Card = 2, and Bank Transfer = 3 may incorrectly imply an ordering.