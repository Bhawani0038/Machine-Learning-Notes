## Topic 8: Data Cleaning

**Objective\:** Combine the techniques learned so far to identify and correct data quality problems before applying encoding, scaling, feature engineering, and machine learning.

Data cleaning is not one operation. It is a structured process involving missing values, duplicate records, incorrect data types, inconsistent categories, and invalid or suspicious values.

## 1\. The data cleaning workflow

**1\. Inspect the dataset**

Rows, columns, data types, statistics

**2\. Remove confirmed duplicates**

Check complete rows and customer IDs

**3\. Correct data types and categories**

Numeric conversion and label standardization

**4\. Investigate invalid values**

Check age, charges, and other business rules

**5\. Handle missing values**

Apply justified imputation or exclusion rules

One distinction is important: deterministic data cleaning can often happen before the train-test split, while operations that *learn parameters from the data*-such as median imputation, scaling, and data-driven outlier thresholds-should be fitted on the training set only.

## 2\. Apply data cleaning to our Customer Churn dataset

The following example starts from the original `df` containing 13 rows. It creates a separate cleaned DataFrame, preserving the original for comparison.

```python

import pandas as pd
import numpy as np

clean_df = df.copy()

# 1. Remove exact duplicate records
clean_df = clean_df.drop_duplicates()

# 2. Convert numerical columns to appropriate types
numeric_columns = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

for column in numeric_columns:
    clean_df[column] = pd.to_numeric(
        clean_df[column],
        errors="coerce"
    )

# 3. Standardize categorical text
categorical_columns = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

for column in categorical_columns:
    clean_df[column] = (
        clean_df[column]
        .astype("string")
        .str.strip()
    )

# Normalize capitalization in gender
clean_df["gender"] = clean_df["gender"].str.capitalize()

# 4. Identify invalid values using business rules
clean_df.loc[
    ~clean_df["age"].between(18, 100),
    "age"
] = np.nan

clean_df.loc[
    clean_df["monthly_charges"] < 0,
    "monthly_charges"
] = np.nan

clean_df.loc[
    ~clean_df["tenure_months"].between(0, 120),
    "tenure_months"
] = np.nan

clean_df.loc[
    clean_df["support_calls"] < 0,
    "support_calls"
] = np.nan

# 5. Check target values
valid_targets = ["Yes", "No"]

clean_df = clean_df[
    clean_df["churn"].isin(valid_targets)
].copy()

# 6. Inspect the cleaned data
print("Shape:", clean_df.shape)
print("\nData types:")
print(clean_df.dtypes)

print("\nMissing values:")
print(clean_df.isna().sum())

print("\nDuplicate rows:", clean_df.duplicated().sum())
```

### Important decisions in this example

- The age range of 18-100 is an illustrative business rule, not a universal rule. Adjust it to the actual customer population.
- We convert invalid ages into missing values rather than deleting the entire customer record.
- We only reject negative charges and counts. We do not automatically remove the ₹50,000 charge because a high charge may be legitimate.
- We retain records with missing input features for now, because we will learn imputation next.
- We validate the target instead of filling a missing target with a guessed label.

## 3\. Inspect the results

Use this code to compare the dataset before and after cleaning.

```python

print("Original shape:", df.shape)
print("Cleaned shape:", clean_df.shape)

print("\nOriginal missing values:")
print(df.isna().sum())

print("\nCleaned missing values:")
print(clean_df.isna().sum())

print("\nCustomer IDs duplicated:")
print(clean_df["customer_id"].duplicated().sum())

print("\nNumerical summary:")
print(clean_df[numeric_columns].describe().round(2))
```

A cleaned dataset does **not** have to contain zero missing values. Sometimes retaining missing values for a later, properly fitted imputer is the correct choice.

Also, do not remove repeated customer IDs automatically: first determine whether each customer should have one row or multiple legitimate records.