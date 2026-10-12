## Topic 6: Incorrect Data Types

**Objective\:** Learn how to identify columns stored with incorrect data types and convert them into appropriate formats before analysis and model training.

We will continue with our Customer Churn dataset.

## 1\. What are data types?

A data type determines how Python and pandas interpret a column's values.

| Data type | Meaning | Example |
|--------|--------|--------|
| `int64` | Integer | `25` |
| `float64` | Decimal number | `1200.50` |
| `object` | Often strings or mixed values | `"Male"` |
| `string` | Text values | `"C001"` |
| `bool` | True or False | `True` |
| `datetime64` | Date and time | `2026-10-12` |
| `category` | Categorical data | `"Monthly"` |

**Important\:** A column can contain numerical-looking values but still be stored as text.

For example, `"24"` is a string, whereas `24` is an integer.

## 2\. Identify incorrect data types

Run\:

```python
print(df.dtypes)
```

Inspect the dataset structure\:

```python
df.info()
```

In our original dataset, `tenure_months` was deliberately converted to strings.

```python
print(df["tenure_months"].head())
print(df["tenure_months"].dtype)
```

Expected data type\:

```text
object
```

This is inappropriate if we want to calculate its mean, standard deviation, or use numerical preprocessing.

## 3\. Convert strings to numerical values

### A. Using `astype()`

Use `astype()` when the values are already valid for the target type.

```python
df["tenure_months"] = df["tenure_months"].astype(int)

print(df["tenure_months"].dtype)
```

Expected output\:

```text
int64
```

However, `astype(int)` raises an error if the column contains missing values or non-numeric strings.

### B. Using `pd.to_numeric()`

For data that may contain invalid values, use `pd.to_numeric()`\.

```python
df["tenure_months"] = pd.to_numeric(
    df["tenure_months"],
    errors="coerce"
)
```

- Valid numerical strings become numbers.
- Invalid strings become `NaN` when `errors="coerce"`\.
- The resulting column can use a floating-point type if missing values exist.

Example\:

```python
sample = pd.Series(["12", "24", "unknown", "36"])

converted = pd.to_numeric(
    sample,
    errors="coerce"
)

print(converted)
```

Expected output\:

```text
0    12.0
1    24.0
2     NaN
3    36.0
dtype: float64
```

**Production practice\:** Inspect the values converted to missing data instead of silently accepting the conversion.

```python
sample = pd.Series(["12", "24", "unknown", "36"])

converted = pd.to_numeric(sample, errors="coerce")

invalid_values = sample[converted.isna()]

print(invalid_values)
```

This identifies `"unknown"` for investigation.

## 4\. Convert text to dates

Real-world datasets frequently store dates as strings.

```python
dates = pd.Series([
    "2026-01-15",
    "2026-02-20",
    "2026-03-10"
])

dates = pd.to_datetime(
    dates,
    errors="coerce"
)

print(dates)
print(dates.dtype)
```

Expected data type\:

```text
datetime64[ns]
```

If a dataset contains dates in formats such as `15/01/2026`, specify the expected format when appropriate\:

```python
dates = pd.to_datetime(
    pd.Series(["15/01/2026", "20/02/2026"]),
    format="%d/%m/%Y",
    errors="coerce"
)
```

This prevents ambiguous date interpretation.

## 5\. Convert numerical values stored with currency symbols

Suppose monthly charges are stored as text\:

```python
charges = pd.Series([
    "₹500",
    "₹1,200",
    "₹2,500"
])
```

Remove the currency symbol and thousands separators before conversion\:

```python
charges = (
    charges
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

charges = pd.to_numeric(
    charges,
    errors="coerce"
)

print(charges)
```

Expected output\:

```text
0     500
1    1200
2    2500
dtype: int64
```

In production, also validate currency conventions, negative amounts, and invalid strings before treating the result as clean financial data.

## 6\. Should every number be converted to a numerical feature?

No. Some numerical-looking columns are identifiers or codes.

| Column | Example | Recommended interpretation |
|--------|--------|--------|
| Customer ID | `10001` | Identifier |
| Postal code | `342001` | Categorical/geographical identifier |
| Rating | `1` to `5` | Ordinal if the scale has meaningful order |
| Number of support calls | `3` | Numerical count |
| Monthly charges | `1200.50` | Numerical |

Converting an identifier into an integer does not make it a useful numerical predictor. The meaning of a column determines how it should be processed.