# 1\. What is Regression?

## 1\. Definition

**Regression is a Supervised Machine Learning technique used to predict a continuous numerical value based on one or more input features.**

- It learns the relationship between input features ($X$\) and a numerical target ($y$\).
- It uses historical data to learn patterns.
- It predicts numerical values for new, unseen data.

## 2\. Real-Life Examples

| Use Case | Input Features (X) | Output (y) |
|--------|--------|--------|
| House Price Prediction | Area, rooms, location | Price: ₹45 lakh |
| Student Marks Prediction | Study hours, attendance | Marks: 85 |
| Salary Prediction | Experience, skills | Salary: ₹8 LPA |
| Sales Prediction | Advertising budget, season | Sales: ₹2,50,000 |
| Temperature Prediction | Humidity, wind speed | Temperature: 32.5°C |

## 3\. How Regression Works

**Input Features (X)**

Study Hours = 6

**Regression Model**

Learns patterns from historical data

**Predicted Output (y)**

## 75 Marks

The predicted marks are illustrative. An actual prediction depends on the data used to train the model.

## 4\. Regression vs Classification

| Regression | Classification |
|--------|--------|
| Predicts continuous numerical values | Predicts categories or classes |
| Output: ₹50,000 | Output: Approved / Rejected |
| Output: 85.5 marks | Output: Pass / Fail |
| Example: Linear Regression | Example: Logistic Regression |

**Important\:** Logistic Regression is generally used for classification despite having the word “Regression” in its name.

## 5\. Simple Mathematical Representation

A regression model can be represented as\:

$$
y = f(X) + \epsilon
$$

Where\:

- $y$ \= Actual target value
- $X$ \= Input features
- $f(X)$ \= Relationship learned by the model
- $\epsilon$ \= Random error or unexplained variation

For simple linear regression\:

$$
\hat{y} = mx + c
$$

Where\:

- $\hat{y}$ \= Predicted value
- $x$ \= Input feature
- $m$ \= Slope or coefficient
- $c$ \= Intercept

## 6\. Python Example

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Input: Study hours
X = np.array([[1], [2], [3], [4], [5]])

# Output: Student marks
y = np.array([20, 30, 40, 50, 60])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 6 study hours
prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])
```

**Output\:**

```text
Predicted Marks: 70.0
```

**Explanation\:**

- `LinearRegression()` creates the model.
- `fit(X, y)` learns the relationship between study hours and marks.
- `predict([[6]])` predicts marks for a student who studies for 6 hours.

This small dataset has a perfectly linear relationship, so the prediction is exact. Real-world datasets generally contain noise and more complex relationships.

## 7\. Key Points to Remember

- Regression is a supervised learning technique.
- It predicts continuous numerical values.
- The target variable is generally numerical.
- Input features can be numerical or categorical after suitable preprocessing.
- Model performance can be evaluated using MAE, MSE, RMSE and $R^2$\.
- Common regression algorithms include Linear Regression, Polynomial Regression, Ridge, Lasso and Decision Tree Regression.