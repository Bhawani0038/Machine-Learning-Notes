# 12\. Lasso Regression

## 1\. What is Lasso Regression?

**Lasso Regression is a regularized version of Linear Regression that uses L1 regularization to reduce overfitting and perform feature selection.**

Lasso stands for **Least Absolute Shrinkage and Selection Operator**\.

For example, suppose you are predicting house prices using these features\:

- Area
- Number of bedrooms
- Number of bathrooms
- Distance from railway station
- House color

If house color does not help predict the price, Lasso may reduce its coefficient to exactly zero, effectively removing that feature from the model.

Key points\:

- Used for regression problems.
- Adds an L1 penalty to the cost function.
- Can make some coefficients exactly zero.
- Helps with feature selection.
- Uses the `alpha` parameter to control regularization.

## 2\. Lasso Regression Formula

Ordinary Linear Regression minimizes the sum of squared errors\:

$$
J(\boldsymbol{\beta})=
\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Lasso adds the absolute values of the coefficients as a penalty\:

$$
\boxed{
J(\boldsymbol{\beta})=
\sum_{i=1}^{n}(y_i-\hat y_i)^2+
\alpha\sum_{j=1}^{p}|\beta_j|
}
$$

Where\:

- $y_i$ \= Actual value
- $\hat y_i$ \= Predicted value
- $\beta_j$ \= Feature coefficient
- $p$ \= Number of features
- $\alpha$ \= Regularization strength

The intercept is not penalized by Scikit-learn's `Lasso` when `fit_intercept=True`\.

## 3\. How Does Lasso Work?

Suppose a model initially learns these coefficients\:

| Feature | Original coefficient | After Lasso (illustrative) |
|--------|--------|--------|
| Area | 25.0 | 23.8 |
| Bedrooms | 8.0 | 6.4 |
| House age | \-3.0 | \-2.1 |
| House color | 0.4 | 0.0 |

Lasso reduces coefficient magnitudes. A less useful feature may end up with a coefficient of exactly zero, meaning the model no longer uses it for prediction.

These numbers are illustrative, not calculated results from a trained model.

## 4\. Understanding the `alpha` Parameter

| Alpha value | Effect |
|--------|--------|
| `0.01` | Very weak regularization |
| `0.1` | Weak regularization |
| `1.0` | Moderate regularization in some datasets |
| `10.0` | Stronger regularization |
| Very large | More coefficients may become zero |

As `alpha` increases, regularization becomes stronger, and the model tends to retain fewer nonzero coefficients. The best value depends on the dataset. citeturn615181search2turn615181search1

## 5\. Complete Python Example

We will reuse the illustrative house-price dataset from the Ridge Regression lesson.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Illustrative dataset
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

# Compare Linear Regression and Lasso
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=1.0, max_iter=10000))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
    print("RMSE:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
    print("R2:", round(r2_score(y_test, predictions), 3))

# Display Lasso coefficients
lasso_model = models["Lasso Regression"].named_steps["model"]

print("\nLasso coefficients:")
for feature, coefficient in zip(X.columns, lasso_model.coef_):
    print(f"{feature}: {coefficient:.4f}")

# Predict a new house price
new_house = pd.DataFrame({
    "area_sqft": [1800],
    "bedrooms": [3],
    "house_age": [8]
})

predicted_price = models["Lasso Regression"].predict(new_house)[0]
print(f"\nPredicted price: ₹{predicted_price:.2f} lakh")
```

**Important\:** The dataset is illustrative, not verified market data. The example standardizes features before applying Lasso because regularization is sensitive to feature scales. The printed coefficients are in standardized-feature units, not the original feature units.

## 6\. Selecting Alpha Automatically

Use `LassoCV` to choose `alpha` through cross-validation rather than manually guessing a value.

```python
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lasso_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCV(
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=5,
        max_iter=10000
    ))
])

lasso_cv.fit(X_train, y_train)

print(
    "Best alpha:",
    lasso_cv.named_steps["model"].alpha_
)
```

## 7\. Ridge vs Lasso Regression

| Feature | Ridge | Lasso |
|--------|--------|--------|
| Regularization | L2 | L1 |
| Penalty | Squared coefficients | Absolute coefficients |
| Shrinks coefficients | Yes | Yes |
| Can set coefficients exactly to zero | Generally no | Yes |
| Feature selection | Indirect | Built-in through zero coefficients |
| Useful when | Many correlated features should be retained | A simpler model with fewer features is desirable |

Both methods can underfit when regularization is too strong.