## Topic 13: Feature Engineering

**Objective\:** Learn how to create new, meaningful features from existing data to help a machine learning model capture useful patterns.

We will continue with our Customer Churn Prediction dataset.

## 1\. What is feature engineering?

Feature engineering is the process of creating, transforming, or combining features using domain knowledge and existing data.

For example, a telecom company may want to understand whether customers with high monthly charges and frequent support calls are more likely to leave.

Instead of using only the original columns, we can derive additional features.

| Existing features | New feature | Purpose |
|--------|--------|--------|
| `monthly_charges`, `tenure_months` | `estimated_total_charges` | Approximate cumulative spending |
| `support_calls` | `high_support_customer` | Flag customers with frequent support calls |
| `monthly_charges` | `charge_category` | Group customers by monthly charge |
| `tenure_months` | `customer_tenure_group` | Group customers by relationship duration |

These are examples for learning. Each feature should be justified by the business problem and available information.

## 2\. Feature engineering using our dataset

### A. Create estimated total charges

Suppose we do not have a reliable historical total-spending column. We can calculate an approximation\:

$$
\text{Estimated Total Charges}
=
\text{Monthly Charges}\times\text{Tenure Months}
$$

```python
df_fe = clean_df.copy()

df_fe["estimated_total_charges"] = (
    df_fe["monthly_charges"] *
    df_fe["tenure_months"]
)

print(
    df_fe[
        [
            "monthly_charges",
            "tenure_months",
            "estimated_total_charges"
        ]
    ].head()
)
```

**Business interpretation\:** This estimates cumulative spending under the assumption that the current monthly charge represents the customer's average charge over the entire tenure.

It is not an actual historical total if prices changed, discounts applied, or the customer had different plans over time.

### B. Create a high-support-call indicator

Suppose the business wants to flag customers who make at least four support calls.

```python
df_fe["high_support_customer"] = (
    df_fe["support_calls"] >= 4
).astype(int)

print(
    df_fe[
        ["support_calls", "high_support_customer"]
    ].head()
)
```

The resulting values are\:

- `1` - four or more support calls.
- `0` - fewer than four support calls.

The threshold of four is illustrative. A real threshold should be based on business requirements, historical patterns, or training-only analysis.

### C. Create customer tenure groups

```python
df_fe["customer_tenure_group"] = pd.cut(
    df_fe["tenure_months"],
    bins=[-1, 12, 36, float("inf")],
    labels=["New", "Established", "Long-term"]
)

print(
    df_fe[
        ["tenure_months", "customer_tenure_group"]
    ]
)
```

The groups represent\:

| Tenure | Group |
|--------|--------|
| 0-12 months | New |
| 13-36 months | Established |
| More than 36 months | Long-term |

Grouping simplifies interpretation, but it also loses some numerical detail. Compare the grouped feature with the original feature before deciding whether to keep both.

### D. Create monthly charge categories

```python
df_fe["charge_category"] = pd.cut(
    df_fe["monthly_charges"],
    bins=[-float("inf"), 800, 1500, float("inf")],
    labels=["Low", "Medium", "High"]
)

print(
    df_fe[
        ["monthly_charges", "charge_category"]
    ].head()
)
```

These thresholds are illustrative, not universal industry standards. Select thresholds based on business meaning rather than choosing them solely to improve training performance.

## 3\. Feature engineering for categorical data

We can combine existing categories to create more meaningful segments.

For example, the business may want to identify customers with both a monthly contract and fiber internet.

```python
df_fe["monthly_fiber_customer"] = (
    (df_fe["contract_type"] == "Monthly") &
    (df_fe["internet_service"] == "Fiber")
).astype(int)

print(
    df_fe[
        [
            "contract_type",
            "internet_service",
            "monthly_fiber_customer"
        ]
    ].head()
)
```

This creates a binary feature indicating whether both conditions are satisfied.

## 4\. Avoid data leakage during feature engineering

Feature engineering must use only information available when the prediction is made.

For example\:

- Valid: Calculate tenure from the customer's joining date and the prediction date.
- Invalid: Use a cancellation date that becomes known only after the customer has already churned.
- Potentially problematic: Calculate charge thresholds using the entire dataset before separating the training and test sets.

A feature can look highly predictive but still be unsuitable if it depends on future information.