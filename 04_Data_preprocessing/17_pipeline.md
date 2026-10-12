# Topic 17 - `Pipeline`

## 1\. What is a Pipeline?

A `Pipeline` in scikit-learn combines multiple preprocessing and machine learning steps into a single object.

For example\:

1. Fill missing numerical values.
2. Scale numerical features.
3. Train a machine learning model.
4. Predict new data.

Instead of running each step manually, we connect them in a pipeline.

## 2\. Without Pipeline

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LogisticRegression()

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
```

Here, we manually manage each preprocessing step.

## 3\. With Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
```

The pipeline automatically performs the steps in order.

## 4\. How does it work?

**Input Features (`X_train`\)**

1\. SimpleImputer - fill missing values

2\. StandardScaler - scale features

3\. LogisticRegression - train model

**Predictions**

## 5\. Important methods

| Method | Purpose |
|--------|--------|
| `fit()` | Learns preprocessing parameters and trains the final estimator |
| `predict()` | Applies the fitted preprocessing steps and generates predictions |
| `score()` | Evaluates the final estimator using its default scoring method |
| `named_steps` | Accesses individual steps by name |

**Important\:** Use `fit()` on training data. When predicting test data, the pipeline reuses the learned preprocessing parameters instead of fitting them again.

---