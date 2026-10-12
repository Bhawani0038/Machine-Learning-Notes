# 09\. Polynomial Regression

## 1\. What is Polynomial Regression?

**Polynomial Regression is a regression technique used to model nonlinear relationships between input features and a continuous target variable by adding polynomial terms to the input features.**

Simple Linear Regression fits a straight line, whereas Polynomial Regression can fit a curved relationship.

Real-life examples\:

- Predicting crop yield based on temperature.
- Modeling vehicle speed versus fuel consumption.
- Predicting product sales as advertising expenditure increases.
- Modeling house prices when price growth is nonlinear with area.

## 2\. Linear vs Polynomial Regression

### Linear Regression
image_group{"query":["simple educational scatter plot with straight upward regression line white background"],"aspect_ratio":"4:3","layout":"carousel","num_per_query":1}

Fits a straight line.

$$
\hat y=b_0+b_1x
$$

### Polynomial Regression
image_group{"query":["simple educational scatter plot with smooth curved polynomial regression line white background"],"aspect_ratio":"4:3","layout":"carousel","num_per_query":1}

Can fit a curved relationship.

$$
\hat y=b_0+b_1x+b_2x^2
$$

Polynomial Regression is still a linear model with respect to its coefficients. The curve comes from transforming the input features.

## 3\. Mathematical Formula

For a polynomial of degree 2\:

$$
\hat y=b_0+b_1x+b_2x^2
$$

For degree 3\:

$$
\hat y=b_0+b_1x+b_2x^2+b_3x^3
$$

Where\:

- $x$ \= Input feature
- $x^2,x^3$ \= Polynomial features
- $b_0$ \= Intercept
- $b_1,b_2,b_3$ \= Learned coefficients

The degree controls the maximum power of the input feature.

| Degree | Polynomial Terms | Typical Shape |
|--------|--------|--------|
| 1 | $x$ | Straight line |
| 2 | $x,x^2$ | Can form a U-shaped or inverted U-shaped curve |
| 3 | $x,x^2,x^3$ | Can model more complex curves |
| 4 | $x,x^2,x^3,x^4$ | Can model still more complex shapes |

A higher degree does not necessarily produce a better model. Excessive complexity can cause overfitting.

## 4\. Python Implementation Using Scikit-learn

We will predict an illustrative target with a curved relationship using `PolynomialFeatures` and `LinearRegression`\.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Input feature
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

# Illustrative nonlinear target
y = np.array([12, 18, 29, 43, 61, 82, 108, 137])

# Create polynomial regression model
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# Train the model
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Evaluate training predictions
mse = mean_squared_error(y, y_pred)

print("Predicted values:", np.round(y_pred, 2))
print("Training MSE:", round(mse, 2))

# Predict for a new input
new_X = np.array([[9]])
print("Prediction for x=9:", round(model.predict(new_X)[0], 2))
```

### Explanation

- `PolynomialFeatures(degree=2)` generates $x$ and $x^2$\.
- `include_bias=False` avoids adding a constant feature because `LinearRegression` already estimates the intercept.
- `make_pipeline()` applies the polynomial transformation before fitting the regression model.
- `fit()` learns the coefficients.
- `predict()` estimates the target for new input values.
- `mean_squared_error()` measures the average squared error.

## 5\. Why Use Polynomial Regression?

Use it when the relationship between the input and target is curved and a straight line does not capture the pattern adequately.

However, always compare the model against a linear baseline using a held-out test set or cross-validation. Training error alone is not enough to establish that a polynomial model generalizes well.