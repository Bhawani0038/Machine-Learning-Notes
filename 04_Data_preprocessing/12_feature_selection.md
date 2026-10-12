## Topic 12: Feature Selection

**Objective\:** Learn how to select relevant input features for machine learning, remove unnecessary columns, reduce redundancy, and avoid selecting features that cause data leakage.

We will continue with our Customer Churn Prediction dataset.

## 1\. What is feature selection?

Feature selection is the process of choosing the most useful input features for predicting a target variable.

Suppose our dataset contains these columns\:

| Feature | Keep or remove? | Reason |
|--------|--------|--------|
| `customer_id` | Usually remove | Identifier, not a meaningful predictor by itself |
| `age` | Consider keeping | May relate to customer behavior |
| `monthly_charges` | Consider keeping | May relate to churn |
| `tenure_months` | Consider keeping | May relate to customer loyalty |
| `support_calls` | Consider keeping | May indicate customer dissatisfaction |
| `gender` | Evaluate carefully | Use only when appropriate and justified |
| `contract_type` | Consider keeping | May be predictive |
| `internet_service` | Consider keeping | May relate to customer needs |
| `payment_method` | Evaluate | May contain useful information |
| `churn` | Target only | This is what we want to predict |

Feature selection can improve interpretability, reduce unnecessary computation, and sometimes improve generalization. Removing features does not automatically improve accuracy.

## 2\. Method 1 - Remove irrelevant columns

The customer ID identifies a record, so we will exclude it from the input features.

```python
X = clean_df.drop(
    columns=["customer_id", "churn", "churn_encoded"],
    errors="ignore"
)

y = clean_df["churn"].map({
    "No": 0,
    "Yes": 1
})

print("Input features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)
```

Here\:

- `X` contains the input features.
- `y` contains the target labels.
- `customer_id` is excluded from the predictors.
- The target is kept separate from `X`\.

Only remove an identifier if it has no justified predictive role. Some identifiers can encode useful grouping or temporal information, but using them directly can also cause overfitting or leakage.

## 3\. Method 2 - Remove constant features

A constant feature has the same value for every observation and usually provides no predictive information.

```python
constant_columns = [
    column
    for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

print("Constant columns:", constant_columns)

X = X.drop(columns=constant_columns)
```

For example, if every customer has the same internet service, that column cannot distinguish between customers in this dataset.

A feature with very low variance may also be unhelpful, but low variance alone does not prove that a feature is useless.

## 4\. Method 3 - Correlation-based selection

Correlation measures the strength and direction of a linear relationship between numerical variables.

```python
numerical_features = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

correlation_matrix = clean_df[
    numerical_features
].corr()

print(correlation_matrix.round(2))
```

A correlation close to `+1` indicates a strong positive linear relationship, while a value close to `-1` indicates a strong negative linear relationship.

A correlation close to `0` indicates little or no linear relationship; a nonlinear relationship may still exist.

### Detect highly correlated feature pairs

```python
import numpy as np

corr = correlation_matrix.abs()

upper_triangle = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)
)

high_correlation_pairs = (
    upper_triangle
    .stack()
    .loc[lambda s: s > 0.90]
)

print(high_correlation_pairs)
```

The threshold `0.90` is an example, not a universal rule.

If two features are highly correlated, investigate whether both are necessary. Do not automatically remove one: correlated predictors can still contribute useful information, depending on the model and business context.

## 5\. Method 4 - Select features using a statistical test

For classification problems, scikit-learn provides `SelectKBest` and `f_classif` for selecting numerical features based on their relationship with the target.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

numerical_features = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

X_num = clean_df[numerical_features]
y = clean_df["churn"].map({"No": 0, "Yes": 1})

# Illustrative example: impute missing values first
imputer = SimpleImputer(strategy="median")
X_num_imputed = imputer.fit_transform(X_num)

selector = SelectKBest(
    score_func=f_classif,
    k=2
)

X_selected = selector.fit_transform(
    X_num_imputed,
    y
)

selected_features = selector.get_feature_names_out(
    numerical_features
)

print("Selected features:", selected_features)
```

This example demonstrates the selection technique, but because it fits preprocessing and selection on the entire dataset, it is **not the correct evaluation workflow** for a real model.

In a production workflow, split the data first and fit the imputer and feature selector on training data only. For reliable performance estimates, perform feature selection inside cross-validation.

`f_classif` evaluates numerical features individually. It does not necessarily identify the best combination of features, detect every nonlinear relationship, or establish causation.

## 6\. Important distinction: Feature selection vs. feature engineering

| Feature selection | Feature engineering |
|--------|--------|
| Chooses existing features | Creates or modifies features |
| Example: retain `age` and `tenure_months` | Example: create `customer_lifetime_value` |
| Removes unnecessary inputs | Adds potentially useful representations |