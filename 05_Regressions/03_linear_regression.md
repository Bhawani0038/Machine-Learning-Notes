# 3\. Linear Regression

## 1\. What is Linear Regression?

**Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value by learning a linear relationship between input features and a target variable.**

Examples\:

- Predicting house prices from house area.
- Predicting student marks from study hours.
- Predicting salary from years of experience.
- Predicting sales from advertising expenditure.

The algorithm finds a best-fit straight line that represents the relationship between input and output values.

## 2\. Mathematical Equation

The equation of Simple Linear Regression is\:

$$
\hat{y}=mx+c
$$

Where\:

| Symbol | Meaning |
|--------|--------|
| $\hat{y}$ | Predicted output |
| $x$ | Input feature |
| $m$ | Slope or coefficient |
| $c$ | Intercept |

Consider this example\:

$$
\text{Marks}=10\times\text{Study Hours}+10
$$

If a student studies for 5 hours\:

$$
\hat{y}=10(5)+10=60
$$

The predicted marks are **60**\.

## 3\. Visual Understanding

### Study Hours vs Predicted Marks

Illustrative linear relationship

The line represents Marks = 10 × Study Hours + 10. This is an illustrative mathematical relationship, not a model trained on real student data.

## 4\. Understanding Slope and Intercept

### A. Slope ($m$\)

The slope tells us how much the predicted output changes when the input increases by one unit.

For the equation\:

$$
\hat{y}=10x+10
$$

The slope is $m=10$\.

Therefore, for every additional study hour, the predicted marks increase by 10.

- Positive slope: output increases as input increases.
- Negative slope: output decreases as input increases.
- Zero slope: predictions remain constant as input changes.

### B. Intercept ($c$\)

The intercept is the predicted output when the input feature equals zero.

In the same equation, $c=10$\.

Therefore, when study hours are zero, the predicted marks are 10.

An intercept is a mathematical model parameter; it does not necessarily represent a realistic real-world outcome.

## 5\. Python Implementation Without Scikit-learn

First, calculate predictions using the equation directly.

```python
study_hours = 5

slope = 10
intercept = 10

predicted_marks = slope * study_hours + intercept

print("Predicted Marks:", predicted_marks)
```

Output\:

```text
Predicted Marks: 60
```

This example uses a manually specified equation. The next example allows Scikit-learn to learn the slope and intercept from data.

## 6\. Linear Regression Using Scikit-learn

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Input feature: study hours
X = np.array([[1], [2], [3], [4], [5]])

# Target: student marks
y = np.array([20, 30, 40, 50, 60])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Display learned parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict marks for 6 study hours
prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])
```

Output\:

```text
Slope: 10.0
Intercept: 10.0
Predicted Marks: 70.0
```

**Explanation of important methods and attributes\:**

- `LinearRegression()` creates the regression model.
- `fit(X, y)` learns the best-fit line from the training data.
- `coef_` provides the learned slope for each feature.
- `intercept_` provides the learned intercept.
- `predict([[6]])` predicts the output for a new input.

## 7\. Limitations of Linear Regression

- It assumes a linear relationship between features and the target.
- Outliers can significantly affect the fitted line.
- Strong multicollinearity can make coefficients unstable in multiple linear regression.
- It may underperform when the true relationship is nonlinear.
- Good training performance does not guarantee good performance on unseen data.