




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

## 6\. Entire Code with Pipe Line & Column Transformer

```python

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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
df['gender'] = df['gender'].str.lower()


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
numeric_transformer

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
categorical_transformer

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
preprocessor
X = df.drop(
    columns=["customer_id", "churn"]
)

y = df["churn"].map({
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
model_pipeline

y_pred = model_pipeline.predict(X_test)

print("Test accuracy:", model_pipeline.score(X_test, y_test))

```


## 7. Code Without Pipeline

```python

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Create the dataset
# --------------------------------------------------

df = pd.DataFrame({
    "customer_id": [
        "C001", "C002", "C003", "C004", "C005", "C006",
        "C007", "C008", "C009", "C010", "C011", "C012"
    ],
    "age": [25, 35, 45, np.nan, 29, 52, 41, 38, 150, 31, 45, 27],
    "monthly_charges": [
        500, 1200, 800, 1500, np.nan, 2200,
        900, 750, 50000, 1100, 800, 650
    ],
    "tenure_months": [
        3, 24, 48, 12, 6, 60, 36, 18, 2, 15, 48, 8
    ],
    "support_calls": [5, 1, 0, 3, 4, 1, 2, 2, 20, 3, 0, 4],
    "gender": [
        "Male", "female", "Male", "Female", None, "Male",
        "Female", "Male", "Female", "Male", "Male", "Female"
    ],
    "contract_type": [
        "Monthly", "Yearly", "Two Year", "Monthly",
        "Monthly", "Two Year", "Yearly", "Monthly",
        "Monthly", "Yearly", "Two Year", "Monthly"
    ],
    "internet_service": [
        "Fiber", "DSL", "Fiber", "None", "DSL", "Fiber",
        "DSL", "Fiber", "Fiber", "DSL", "Fiber", "None"
    ],
    "payment_method": [
        "UPI", "Card", "Bank Transfer", "UPI", "Card",
        "Bank Transfer", "UPI", "Card", "UPI", "Card",
        "Bank Transfer", "UPI"
    ],
    "churn": [
        "Yes", "No", "No", "Yes", "Yes", "No",
        "No", "No", "Yes", "No", "No", "Yes"
    ]
})

# --------------------------------------------------
# 2. Clean the dataset
# --------------------------------------------------

df = df.drop_duplicates().copy()

# Correct the numerical column's data type
df["tenure_months"] = pd.to_numeric(
    df["tenure_months"], errors="coerce"
)

# Standardize categorical text
df["gender"] = df["gender"].str.lower().str.strip()

# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["customer_id", "churn"])

y = df["churn"].map({
    "No": 0,
    "Yes": 1
})

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

# --------------------------------------------------
# 4. Split before fitting preprocessing objects
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 5. Handle missing numerical values
# --------------------------------------------------

numeric_imputer = SimpleImputer(strategy="median")

X_train_num = pd.DataFrame(
    numeric_imputer.fit_transform(
        X_train[numeric_features]
    ),
    columns=numeric_features,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    numeric_imputer.transform(
        X_test[numeric_features]
    ),
    columns=numeric_features,
    index=X_test.index
)

# --------------------------------------------------
# 6. Scale numerical features
# --------------------------------------------------

scaler = StandardScaler()

X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train_num),
    columns=numeric_features,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    scaler.transform(X_test_num),
    columns=numeric_features,
    index=X_test.index
)

# --------------------------------------------------
# 7. Handle missing categorical values
# --------------------------------------------------

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

X_train_cat = pd.DataFrame(
    categorical_imputer.fit_transform(
        X_train[categorical_features]
    ),
    columns=categorical_features,
    index=X_train.index
)

X_test_cat = pd.DataFrame(
    categorical_imputer.transform(
        X_test[categorical_features]
    ),
    columns=categorical_features,
    index=X_test.index
)

# --------------------------------------------------
# 8. Encode categorical features
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded = pd.DataFrame(
    encoder.fit_transform(X_train_cat),
    columns=encoder.get_feature_names_out(
        categorical_features
    ),
    index=X_train.index
)

X_test_encoded = pd.DataFrame(
    encoder.transform(X_test_cat),
    columns=encoder.get_feature_names_out(
        categorical_features
    ),
    index=X_test.index
)

# --------------------------------------------------
# 9. Combine numerical and categorical features
# --------------------------------------------------

X_train_final = pd.concat(
    [X_train_num, X_train_encoded], axis=1
)

X_test_final = pd.concat(
    [X_test_num, X_test_encoded], axis=1
)

# --------------------------------------------------
# 10. Train the model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_final, y_train)

# --------------------------------------------------
# 11. Make predictions and evaluate
# --------------------------------------------------

y_pred = model.predict(X_test_final)

print("Actual values:", y_test.to_list())
print("Predicted values:", y_pred)
print("Test accuracy:", model.score(X_test_final, y_test))
```