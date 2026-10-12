# 11\. Ridge Regression

## 1\. What is Ridge Regression?

**Ridge Regression is a regularized version of Linear Regression that adds an L2 penalty to the cost function to discourage excessively large coefficients.**

It is useful when a model has many input features, correlated features, or a tendency to overfit.

For example, when predicting house prices, features such as built-up area, number of rooms, and number of bathrooms may be correlated. Ridge Regression can help stabilize the learned coefficients.

Key points\:

- It is used for regression problems.
- It penalizes large coefficients.
- It can reduce overfitting.
- It generally shrinks coefficients toward zero but does not usually make them exactly zero.
- Its regularization strength is controlled by `alpha`\.

## 2\. Linear Regression vs Ridge Regression

Ordinary Linear Regression minimizes the sum of squared errors\:

$$
J(\boldsymbol{\beta})=
\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Ridge Regression adds an L2 penalty\:

$$
\boxed{
J(\boldsymbol{\beta})=
\sum_{i=1}^{n}(y_i-\hat y_i)^2+
\alpha\sum_{j=1}^{p}\beta_j^2
}
$$

Where\:

- $y_i$ \= Actual target value
- $\hat y_i$ \= Predicted target value
- $\beta_j$ \= Coefficient for feature $j$
- $p$ \= Number of features
- $\alpha$ \= Regularization strength

The intercept is not penalized by Scikit-learn's `Ridge` when `fit_intercept=True`\.

## 3\. Understanding the `alpha` Parameter

The `alpha` parameter controls how strongly large coefficients are penalized.

| Alpha | Effect |
|--------|--------|
| `0` | No regularization; use `LinearRegression` for ordinary least squares |
| `0.01` | Weak regularization |
| `1` | Moderate regularization in many practical settings |
| `100` | Stronger regularization |
| Extremely large | Coefficients are pushed strongly toward zero |

The actual effect depends on the scale of the features and target, the dataset size, and the magnitude of the original coefficients.

**Important\:** Excessive regularization can cause underfitting. The best alpha should be selected using cross-validation rather than assumed in advance.

## 4\. Complete Python Example

We will use a house-price dataset and compare ordinary Linear Regression with Ridge Regression.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Create models with feature scaling
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
    print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
    print("R2:", round(r2_score(y_test, predictions), 3))

# Predict a new house price using Ridge
new_house = pd.DataFrame({
    "area_sqft": [1800],
    "bedrooms": [3],
    "house_age": [8]
})

ridge_model = models["Ridge Regression"]
ridge_model.fit(X_train, y_train)

price = ridge_model.predict(new_house)[0]
print(f"\nPredicted price: ₹{price:.2f} lakh")
```

The dataset is illustrative, not verified market data. Compare the held-out test metrics to determine which model performs better; Ridge is not guaranteed to win on every dataset.

## 5\. Selecting Alpha with Cross-Validation

Instead of choosing `alpha=1.0` arbitrarily, use `RidgeCV` to select a value from a candidate range.

```python
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeCV(
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0]
    ))
])

ridge_cv.fit(X_train, y_train)

print(
    "Best alpha:",
    ridge_cv.named_steps["model"].alpha_
)
```

`RidgeCV` performs cross-validation to choose the regularization strength. With the default settings, it uses efficient leave-one-out cross-validation. For larger workflows, an explicitly chosen `cv` strategy and a pipeline inside `GridSearchCV` can be useful.

## 6\. Ridge vs Linear Regression

| Linear Regression | Ridge Regression |
|--------|--------|
| Minimizes squared errors | Minimizes squared errors plus L2 penalty |
| No regularization | Uses regularization |
| Coefficients can become large | Coefficients are shrunk |
| May be unstable with highly correlated features | Often improves coefficient stability |
| No `alpha` parameter | Has an `alpha` parameter |