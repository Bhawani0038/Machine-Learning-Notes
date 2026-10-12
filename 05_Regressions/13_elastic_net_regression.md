# 13\. Elastic Net Regression

## 1\. What is Elastic Net Regression?

**Elastic Net Regression combines Ridge Regression (L2 regularization) and Lasso Regression (L1 regularization).**

It helps control overfitting while allowing some feature coefficients to become exactly zero.

For example, when predicting house prices, you might have many related features\:

- Area in square feet
- Number of rooms
- Number of bedrooms
- Number of bathrooms
- Distance from the railway station

Elastic Net can shrink coefficients and remove some less useful features, making it helpful when many features are correlated.

## 2\. Elastic Net Formula

Elastic Net combines the penalties from Ridge and Lasso\:

$$
\boxed{
J(\boldsymbol{\beta}) =
\sum_{i=1}^{n}(y_i-\hat y_i)^2
+\alpha\rho\sum_{j=1}^{p}|\beta_j|
+\frac{\alpha(1-\rho)}{2}
\sum_{j=1}^{p}\beta_j^2
}
$$

Where\:

- $y_i$ \= Actual value
- $\hat y_i$ \= Predicted value
- $\beta_j$ \= Model coefficient
- $\alpha$ \= Overall regularization strength
- $\rho$ \= Balance between Lasso and Ridge penalties

In Scikit-learn, the balance parameter is named `l1_ratio`, corresponding to $\rho$\.

## 3\. Understanding `alpha` and `l1_ratio`

| `l1_ratio` | Regularization behavior |
|--------|--------|
| `0` | Ridge-like L2 regularization |
| `0.25` | More L2, less L1 |
| `0.5` | Equal L1 and L2 penalty weights |
| `0.75` | More L1, less L2 |
| `1` | Lasso-like L1 regularization |

The `alpha` parameter controls the overall strength of regularization\:

- Smaller `alpha` → weaker regularization.
- Larger `alpha` → stronger regularization.

Note: `ElasticNet` with `l1_ratio=0` is mathematically Ridge-like, but Scikit-learn recommends using `Ridge` directly for pure L2 regularization.

## 4\. Complete Python Example

We will compare Linear Regression, Ridge, Lasso, and Elastic Net using the same illustrative house-price dataset.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# Illustrative house-price dataset
df = pd.DataFrame({
    "area_sqft": [
        650, 720, 800, 850, 900, 950, 1000, 1050, 1100, 1150,
        1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650,
        1700, 1750, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2300,
        2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300
    ],
    "bedrooms": [
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 5, 5, 5
    ],
    "house_age": [
        18, 12, 8, 20, 15, 10, 6, 22, 13, 9,
        5, 17, 11, 19, 14, 7, 4, 16, 10, 6,
        21, 12, 8, 5, 18, 9, 15, 7, 11, 4,
        13, 6, 10, 3, 8, 5, 2, 12, 4, 1
    ],
    "price_lakh": [
        22, 25, 29, 31, 33, 36, 39, 38, 42, 44,
        48, 47, 51, 54, 57, 61, 65, 64, 69, 72,
        74, 78, 82, 85, 87, 91, 96, 101, 106, 111,
        116, 122, 127, 132, 138, 143, 149, 154, 160, 166
    ]
})

X = df[["area_sqft", "bedrooms", "house_age"]]
y = df["price_lakh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=1.0, max_iter=10000),

    "Elastic Net": ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=10000
    )
}

for name, estimator in models.items():

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print(
        "MAE:",
        round(mean_absolute_error(y_test, predictions), 2)
    )
    print(
        "RMSE:",
        round(mean_squared_error(y_test, predictions) ** 0.5, 2)
    )
    print(
        "R2:",
        round(r2_score(y_test, predictions), 3)
    )
```

The dataset is illustrative, not verified market data. Compare the test metrics to determine which model works best; Elastic Net is not guaranteed to outperform the others.

## 5\. Selecting the Best Parameters

Use `ElasticNetCV` to select `alpha` and `l1_ratio` through cross-validation.

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

elastic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(
        alphas=[0.01, 0.1, 1.0, 10.0],
        l1_ratio=[0.2, 0.5, 0.8, 1.0],
        cv=5,
        max_iter=10000
    ))
])

elastic_model.fit(X_train, y_train)

best_model = elastic_model.named_steps["model"]

print("Best alpha:", best_model.alpha_)
print("Best l1_ratio:", best_model.l1_ratio_)
```

## 6\. Ridge vs Lasso vs Elastic Net

| Feature | Ridge | Lasso | Elastic Net |
|--------|--------|--------|--------|
| Penalty | L2 | L1 | L1 + L2 |
| Shrinks coefficients | Yes | Yes | Yes |
| Can set coefficients to zero | Generally no | Yes | Yes |
| Handles correlated features | Often useful | May select one among correlated features | Often useful |
| Main advantage | Coefficient stability | Feature selection | Combines both approaches |

**When to use Elastic Net\:** Consider it when you have many input features, especially correlated ones, and want both coefficient shrinkage and potential feature selection.

---