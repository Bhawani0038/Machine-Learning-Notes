



## Topic 1: Understanding the Dataset

**Objective\:** Before cleaning or transforming data, learn how to inspect a dataset, understand its structure, identify data quality problems, and determine which columns will be used for machine learning.

We will use the same Customer Churn Prediction dataset throughout this module.

### 1\. Load and inspect the dataset

Run the following code in VS Code, Jupyter Notebook, or Google Colab.

```python

import pandas as pd
import numpy as np

df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003", "C004", "C005",
                    "C006", "C007", "C008", "C009", "C010",
                    "C011", "C012"],
    "age": [25, 35, 45, np.nan, 29, 52, 41, 38, 150, 31, 45, 27],
    "monthly_charges": [500, 1200, 800, 1500, np.nan, 2200,
                        900, 750, 50000, 1100, 800, 650],
    "tenure_months": [3, 24, 48, 12, 6, 60, 36, 18, 2, 15, 48, 8],
    "support_calls": [5, 1, 0, 3, 4, 1, 2, 2, 20, 3, 0, 4],
    "gender": ["Male", "female", "Male", "Female", None, "Male",
               "Female", "Male", "Female", "Male", "Male", "Female"],
    "contract_type": ["Monthly", "Yearly", "Two Year", "Monthly",
                      "Monthly", "Two Year", "Yearly", "Monthly",
                      "Monthly", "Yearly", "Two Year", "Monthly"],
    "internet_service": ["Fiber", "DSL", "Fiber", "None", "DSL",
                         "Fiber", "DSL", "Fiber", "Fiber", "DSL",
                         "Fiber", "None"],
    "payment_method": ["UPI", "Card", "Bank Transfer", "UPI", "Card",
                       "Bank Transfer", "UPI", "Card", "UPI", "Card",
                       "Bank Transfer", "UPI"],
    "churn": ["Yes", "No", "No", "Yes", "Yes", "No",
              "No", "No", "Yes", "No", "No", "Yes"]
})

# Add a duplicate record
df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

# Introduce a string data type for a numerical column
df["tenure_months"] = df["tenure_months"].astype(str)

# Inspect the dataset
print(df.head())
```

### 2\. Essential inspection methods

Run each command separately and observe the output.

**A. View the first five rows**

```python
df.head()
```

Checks column names, sample values, missing entries, and category inconsistencies.

---

**B. Check the dataset dimensions**

```python
df.shape
```

Expected: (13, 10) - 13 rows and 10 columns, including the duplicate row.

---

**C. Check column data types and non-null counts**

```python
df.info()
```

Reveals missing values and that `tenure_months` is currently an object/string column.

---

**D. Generate descriptive statistics**

```python
df.describe()
```

Shows count, mean, standard deviation, minimum, quartiles, and maximum for numerical columns.

---

**E. Inspect every column's data type**

```python
df.dtypes
```

Helps identify columns stored using inappropriate data types.

---

**F. Count missing values**

```python
df.isnull().sum()
```

Identifies columns requiring missing-value treatment.

---

**G. Count duplicate rows**

```python
df.duplicated().sum()
```

Expected: 1 duplicate row.

---

**H. Inspect unique categorical values**

```python
print(df["gender"].unique())
print(df["contract_type"].unique())
print(df["payment_method"].unique())
```

Helps find inconsistent labels such as `Male` and `female`\.

### 3\. Numerical vs. categorical features

| Feature type | Columns | What to investigate |
|--------|--------|--------|
| Numerical | `age`, `monthly_charges`, `support_calls` | Missing values, ranges, outliers |
| Numerical stored as text | `tenure_months` | Incorrect data type |
| Categorical | `gender`, `contract_type`, `internet_service`, `payment_method` | Missing values, inconsistent labels |
| Identifier | `customer_id` | Uniqueness and suitability for removal |
| Target | `churn` | Class distribution and target validity |

Notice that `tenure_months` contains numbers, but its data type is `object`\. This is why understanding the dataset must come before applying scaling or model training.