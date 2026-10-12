# 15\. Regression Model Comparison

## 1\. What is Regression Model Comparison?

Regression model comparison means training multiple regression algorithms on the same dataset and comparing their performance to select a suitable model.

For example, when predicting house prices, we might compare\:

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Decision Tree Regression
- Random Forest Regression

## 2\. Common Regression Evaluation Metrics

| Metric | Meaning | Better result |
|--------|--------|--------|
| MAE | Average absolute prediction error | Lower |
| MSE | Average squared prediction error | Lower |
| RMSE | Square root of MSE | Lower |
| R² | Performance relative to predicting the target mean | Generally higher |

Do not compare models using only one metric. Consider the problem, the cost of prediction errors, and performance on unseen data.

## 3\. Complete Python Code to Compare Models

The following example uses the existing house-price dataset and compares four models.

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Features and target
X = df[["area_sqft", "bedrooms", "house_age"]]
y = df["price_lakh"]

# Split data once so every model uses the same test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Polynomial Regression": Pipeline([
        ("polynomial", PolynomialFeatures(
            degree=2,
            include_bias=False
        )),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=1.0, max_iter=10000))
    ]),

    "Elastic Net": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=10000
        ))
    ])
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    })

results_df = pd.DataFrame(results)

# Sort by test RMSE, from lowest to highest
results_df = results_df.sort_values("RMSE")

print(results_df.round(3).to_string(index=False))
```

This example assumes `df` contains the `area_sqft`, `bedrooms`, `house_age`, and `price_lakh` columns used in the earlier house-price lessons.

## 4\. How to Select the Best Model

Use the comparison table to examine\:

- **Lowest MAE\:** Typically indicates smaller average absolute errors.
- **Lowest RMSE\:** Favors models that avoid large errors.
- **Highest R²\:** Indicates a stronger fit relative to the target mean on that test set.
- **Generalization\:** Check whether the model performs well on unseen data and is not excessively complex.

A model with the best test score is not automatically the best final choice. For a reliable comparison, use cross-validation for model selection and keep a separate test set for final evaluation.

