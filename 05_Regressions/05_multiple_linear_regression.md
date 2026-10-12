# 5\. Multiple Linear Regression

## 1\. What is Multiple Linear Regression?

**Multiple Linear Regression is a supervised Machine Learning algorithm that predicts a continuous numerical value using two or more input features.**

Unlike Simple Linear Regression, which uses only one input feature, Multiple Linear Regression considers multiple factors to make a prediction.

**Real-life example: House Price Prediction**

A house price may depend on\:

- Area in square feet
- Number of bedrooms
- Number of floors
- Car parking availability

The model learns how these features relate to the house price.

## 2\. Mathematical Formula

The general equation is\:

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

Where\:

| Symbol | Meaning |
|--------|--------|
| $\hat y$ | Predicted output |
| $b_0$ | Intercept |
| $b_1,b_2,\ldots,b_n$ | Learned coefficients |
| $x_1,x_2,\ldots,x_n$ | Input features |
| $n$ | Number of input features |

For example\:

$$
\begin{aligned}
\text{Price (₹ lakh)}={}&10+0.04(\text{Area})\\
&+5(\text{Bedrooms})\\
&+3(\text{Parking})
\end{aligned}
$$

Suppose a house has an area of 1,000 sq. ft., 3 bedrooms, and parking available ($1$\).

$$
\text{Price}=10+0.04(1000)+5(3)+3(1)
$$

$$
\text{Price}=68\text{ lakh}
$$

This is an illustrative equation with assumed coefficients, not a model trained on actual real-estate data.

## 3\. Complete Python Implementation

We will create a small house-price dataset, train a Multiple Linear Regression model, and predict the price of a new house.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample training dataset
df = pd.DataFrame({
    "area_sqft": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "bedrooms":  [2,   2,    3,    3,    3,    4,    4,    4],
    "parking":   [0,   1,    1,    0,    1,    1,    0,    1],
    "price_lakh": [30, 38,   45,   48,   58,   65,   68,   78]
})

# Separate input features and target
X = df[["area_sqft", "bedrooms", "parking"]]
y = df["price_lakh"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a new house
new_house = pd.DataFrame({
    "area_sqft": [1600],
    "bedrooms": [3],
    "parking": [1]
})

predicted_price = model.predict(new_house)

print("Predicted Price:", round(predicted_price[0], 2), "Lakh")

# Inspect learned parameters
print("Intercept:", round(model.intercept_, 2))
print("Coefficients:", model.coef_)

# Training-set evaluation for demonstration
y_pred = model.predict(X)

print("Training MAE:", round(mean_absolute_error(y, y_pred), 2))
print("Training R2:", round(r2_score(y, y_pred), 3))
```

### Understanding the code

- `X` contains three independent variables.
- `y` contains the target variable, `price_lakh`\.
- `fit()` learns the coefficient for each feature and the intercept.
- `predict()` estimates the price of a new house.
- `coef_` shows how the prediction changes with each feature when the other features are held constant.
- `MAE` measures the average absolute prediction error.

The dataset is small and illustrative. The training metrics do not establish how well the model predicts unseen house prices.

## 4\. Simple vs Multiple Linear Regression

| Basis | Simple Linear Regression | Multiple Linear Regression |
|--------|--------|--------|
| Input features | One | Two or more |
| Example | Price from area | Price from area, bedrooms and parking |
| Equation | $\hat y=b_0+b_1x_1$ | $\hat y=b_0+\sum_{j=1}^{n}b_jx_j$ |
| Visualization | A straight line in 2D | A plane for two features; higher-dimensional relationship for more |

## 5\. Important Points

- Multiple Linear Regression predicts continuous numerical values.
- Each coefficient represents the feature's estimated contribution, holding other included features constant.
- Correlated input features can make coefficients unstable.
- Outliers and nonlinear relationships can affect performance.
- Evaluate generalization using an appropriate train-test split or cross-validation.

---