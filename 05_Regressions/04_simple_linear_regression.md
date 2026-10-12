# 4\. Simple Linear Regression

## 1\. What is Simple Linear Regression?

**Simple Linear Regression is a supervised Machine Learning algorithm that predicts a continuous numerical output using only one input feature.**

It assumes that the relationship between the input and output can be approximated by a straight line.

Examples\:

- Predicting marks using study hours.
- Predicting salary using years of experience.
- Predicting house price using area.
- Predicting electricity consumption using temperature.

**Key point\:** Simple Linear Regression uses one independent variable ($X$\) and one dependent variable ($y$\).

## 2\. Mathematical Formula

$$
\hat{y}=mx+c
$$

Where\:

- $\hat{y}$\: predicted output
- $x$\: single input feature
- $m$\: slope or coefficient
- $c$\: intercept

Consider a salary prediction example\:

$$
\text{Salary}=5000x+20000
$$

Here, $x$ represents years of experience and salary is measured in rupees per month.

If experience is 4 years\:

$$
\hat{y}=5000(4)+20000
$$

$$
\hat{y}=₹40,000
$$

This equation predicts a monthly salary of ₹40,000.

## 3\. Understanding Actual vs Predicted Values

Suppose we have the following training data\:

| Study Hours (X) | Actual Marks (y) |
|--------|--------|
| 1 | 22 |
| 2 | 28 |
| 3 | 39 |
| 4 | 42 |
| 5 | 53 |

The model learns a line that minimizes the sum of squared residuals on the training data.

A **residual** is the difference between an actual value and its predicted value\:

$$
e_i=y_i-\hat{y}_i
$$

For example, if actual marks are 42 and predicted marks are 44\:

$$
e=42-44=-2
$$

The model overpredicted by 2 marks.

## 4\. Complete Python Implementation

We will train a Simple Linear Regression model using Scikit-learn, make predictions, and evaluate its performance.

```python
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Training data
df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5],
    "marks": [22, 28, 39, 42, 53]
})

# Separate input and target
X = df[["study_hours"]]
y = df["marks"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for new study hours
new_data = pd.DataFrame({"study_hours": [6, 7]})
predictions = model.predict(new_data)

print("Slope:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))

for hours, marks in zip(new_data["study_hours"], predictions):
    print(f"{hours} hours -> Predicted marks: {marks:.2f}")

# Evaluate on the training data for demonstration
y_pred = model.predict(X)

print("MAE:", round(mean_absolute_error(y, y_pred), 2))
print("MSE:", round(mean_squared_error(y, y_pred), 2))
print("R2 Score:", round(r2_score(y, y_pred), 3))
```

### Code explanation

- `X = df[["study_hours"]]` selects one feature as a two-dimensional DataFrame.
- `y = df["marks"]` selects the target variable.
- `model.fit(X, y)` learns the slope and intercept.
- `model.predict(new_data)` predicts marks for new study-hour values.
- `MAE` measures the average absolute prediction error.
- `MSE` measures the average squared prediction error.
- `R2 Score` measures how much of the target's variation the model explains on the evaluated data.

The metrics above evaluate the training data, so they should not be treated as evidence of performance on unseen students.

## 5\. Visualize the Regression Line

You can plot the actual observations and the fitted line using Matplotlib.

```python
import matplotlib.pyplot as plt

plt.scatter(X["study_hours"], y, label="Actual Marks")

plt.plot(
    X["study_hours"],
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
```

The scatter points show actual observations, while the line shows the values predicted by the trained model.

## 6\. Simple vs Multiple Linear Regression

| Feature | Simple Linear Regression | Multiple Linear Regression |
|--------|--------|--------|
| Number of input features | One | Two or more |
| Equation | $\hat y=mx+c$ | $\hat y=b_0+b_1x_1+\cdots+b_nx_n$ |
| Example | Salary from experience | Salary from experience, skills and education |
| Complexity | Generally simpler | Depends on features and relationships |