




# Topic 18 - `ColumnTransformer`

## 1\. What is ColumnTransformer?

`ColumnTransformer` applies different preprocessing techniques to different columns of a dataset.

For example, in Customer Churn Prediction\:

- Numerical columns need missing-value imputation and scaling.
- Categorical columns need missing-value imputation and one-hot encoding.

A single `StandardScaler` cannot appropriately process raw text categories such as `"Fiber"` and `"DSL"`\.

## 2\. Example dataset

| age | monthly\_charges | contract\_type | internet\_service |
|--------|--------|--------|--------|
| 25 | 500 | Monthly | Fiber |
| 35 | 1200 | Yearly | DSL |
| NaN | 800 | Two Year | Fiber |

We need different transformations for the numerical and categorical columns.

## 3\. ColumnTransformer code

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

categorical_features = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
```

### What happens here?

| Component | Responsibility |
|--------|--------|
| `numeric_features` | Selects numerical columns |
| `categorical_features` | Selects categorical columns |
| `numeric_transformer` | Imputes missing values and scales numbers |
| `categorical_transformer` | Imputes missing values and encodes categories |
| `preprocessor` | Applies the appropriate transformer to each column |

## 4\. Combine Pipeline and ColumnTransformer

This is the common approach used in real-world machine learning projects.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = clean_df.drop(
    columns=["customer_id", "churn"]
)

y = clean_df["churn"].map({
    "No": 0,
    "Yes": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

print("Test accuracy:", model_pipeline.score(X_test, y_test))
```

This assumes `clean_df` contains the cleaned columns defined above and that both churn classes have enough observations for a stratified split.

Notice that `customer_id` is excluded because it is an identifier rather than a meaningful predictive feature. The target `churn` is also excluded from `X` to prevent target leakage.

## 5\. Pipeline vs. ColumnTransformer

| Pipeline | ColumnTransformer |
|--------|--------|
| Connects sequential steps | Applies transformations to selected columns |
| Can include preprocessing and a model | Primarily organizes column-specific preprocessing |
| Example: imputer → scaler → model | Example: scale numerical columns, encode categorical columns |
| Commonly wraps a `ColumnTransformer` | Commonly used inside a `Pipeline` |

**Remember\:** `Pipeline` answers *“In what order should the steps run?”* `ColumnTransformer` answers *“Which transformation should be applied to which columns?”*

