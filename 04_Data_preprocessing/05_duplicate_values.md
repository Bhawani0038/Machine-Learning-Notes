## Topic 5: Duplicate Values

**Objective\:** Learn to detect duplicate rows, identify repeated IDs, decide which records to remove, and understand how duplicates can affect machine learning models.

We will use the same Customer Churn dataset.

## 1\. What are duplicate values?

Duplicates occur when a row or a key value appears more than once.

There are two important cases\:

- **Duplicate rows\:** Two or more records have identical values across all columns.
- **Duplicate IDs\:** Two or more records share the same identifier, even if their other values differ.

Example\:

| customer\_id | age | monthly\_charges | churn |
|--------|--------|--------|--------|
| C001 | 25 | 500 | Yes |
| C002 | 35 | 1200 | No |
| C003 | 45 | 800 | No |
| C003 | 45 | 800 | No |
| C004 | 29 | 1500 | Yes |

Here, the two `C003` records are exact duplicates.

## 2\. Detect duplicate rows

### A. Count duplicate rows

```python
print("Duplicate rows:", df.duplicated().sum())
```

The expected count for our original practice dataset is `1`, because we deliberately appended one duplicate record.

By default, `duplicated()` marks subsequent occurrences as duplicates and treats the first occurrence as the original.

### B. Display duplicate rows

```python
duplicate_rows = df[df.duplicated()]

print(duplicate_rows)
```

### C. Display every occurrence of duplicated rows

```python
duplicate_rows = df[
    df.duplicated(keep=False)
]

print(duplicate_rows)
```

The `keep=False` argument marks all occurrences of each duplicated row, including the first one.

## 3\. Remove exact duplicate rows

Use `drop_duplicates()` to remove repeated records.

```python
df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))
```

Our dataset will contain 12 rows after removing the single duplicate.

By default, pandas retains the first occurrence.

**Important\:** Do not remove duplicates automatically without understanding the data. Two customers can have identical ages, charges, and churn labels but still be different customers if their IDs differ.

## 4\. Detect duplicate customer IDs

Sometimes records are not identical, but their IDs repeat.

```python
duplicate_ids = df[
    df.duplicated(
        subset=["customer_id"],
        keep=False
    )
]

print(duplicate_ids)
```

This checks duplicates based only on `customer_id`\.

Count the number of repeated ID records\:

```python
print(
    df["customer_id"].duplicated().sum()
)
```

If customer IDs are supposed to be unique, repeated IDs should be investigated. They may represent duplicate imports, multiple snapshots, or legitimate transactions associated with one customer.

## 5\. Remove duplicates based on selected columns

Suppose we want only one record per customer ID.

```python
df = df.drop_duplicates(
    subset=["customer_id"],
    keep="first"
)
```

This retains the first row for each ID and removes later rows sharing that ID.

However, this is appropriate only when the business rules establish that one row per customer is required and the retained record is the correct one.

If records have timestamps, a better approach might be to retain the latest valid record.

For example, if a dataset has a `record_updated_at` column\:

```python
df = (
    df.sort_values("record_updated_at")
      .drop_duplicates(
          subset=["customer_id"],
          keep="last"
      )
)
```

This retains the latest timestamp for each customer, assuming the timestamps and update rules are trustworthy.

## 6\. Why do duplicates matter in machine learning?

Duplicates can cause several problems\:

- **Biased evaluation\:** Duplicate records may appear in both training and test sets, making model performance look better than it really is.
- **Overrepresented observations\:** Repeated records can give certain customers disproportionate influence.
- **Incorrect business metrics\:** Customer counts and churn rates may be inaccurate.
- **Misleading statistics\:** Means, distributions, and category frequencies can change.

Removing duplicates before splitting data is often appropriate when the duplicates are confirmed ingestion errors. For time-dependent or grouped data, deduplication and splitting must follow the underlying business rules.