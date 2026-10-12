# 6\. Cost Function in Machine Learning

## 1\. What is a Cost Function?

A **Cost Function** measures how far a Machine Learning model's predictions are from the actual target values.

It helps us evaluate how well a model fits the training data. During training, an optimization algorithm can use the cost function to find model parameters that reduce prediction errors.

**Example\:** Predicting student marks.

| Study Hours | Actual Marks | Predicted Marks | Error |
|--------|--------|--------|--------|
| 1 | 20 | 25 | 5 |
| 2 | 30 | 32 | 2 |
| 3 | 40 | 38 | \-2 |
| 4 | 50 | 55 | 5 |

The error is calculated as\:

$$
e_i=y_i-\hat{y}_i
$$

Where\:

- $y_i$ \= Actual value
- $\hat{y}_i$ \= Predicted value
- $e_i$ \= Prediction error or residual

A cost function combines the errors across multiple observations into a single number.

## 2\. Why Do We Need a Cost Function?

Suppose two models predict marks for the same students.

| Student | Actual | Model A | Model B |
|--------|--------|--------|--------|
| A | 50 | 48 | 40 |
| B | 60 | 62 | 75 |
| C | 70 | 69 | 65 |

Model A generally predicts closer to the actual values. A cost function allows us to quantify this difference rather than relying on visual inspection.

Cost functions are useful for\:

- Comparing model predictions.
- Optimizing model parameters.
- Training linear regression with gradient-based methods.
- Monitoring whether training error is decreasing.

## 3\. Mean Squared Error (MSE) as a Cost Function

A commonly used cost function for regression is Mean Squared Error (MSE).

$$
\boxed{MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Where\:

- $n$ \= Number of observations
- $y_i$ \= Actual value
- $\hat{y}_i$ \= Predicted value

Using the first table, the errors are $5, 2, -2, 5$\.

$$
MSE=\frac{5^2+2^2+(-2)^2+5^2}{4}
$$

$$
MSE=\frac{25+4+4+25}{4}
=\boxed{14.5}
$$

The MSE is 14.5 marks squared.

**Why square the errors?**

- Negative and positive errors do not cancel each other out.
- Larger errors receive greater penalties.
- The resulting function is mathematically convenient for optimization in linear regression.

## 4\. Interactive Example: Calculate MSE

Change the actual and predicted marks to see how the cost changes.

---

Mean Squared Error

[Reset values]

## 5\. Python Implementation

```python
from sklearn.metrics import mean_squared_error

actual = [50, 60, 70]
predicted = [48, 62, 69]

mse = mean_squared_error(actual, predicted)

print("MSE:", mse)
```

Output\:

```text
MSE: 3.0
```

Calculation\:

$$
MSE=\frac{(50-48)^2+(60-62)^2+(70-69)^2}{3}
$$

$$
MSE=\frac{4+4+1}{3}=\boxed{3}
$$

## 6\. Cost Function vs Evaluation Metric

These concepts overlap, but their roles differ.

| Cost Function | Evaluation Metric |
|--------|--------|
| Often used to optimize model parameters | Used to assess model performance |
| MSE can be minimized during training | MAE, RMSE and $R^2$ can assess predictions |
| Depends on the training objective | Depends on the evaluation goal |

For ordinary Linear Regression in Scikit-learn, `LinearRegression()` minimizes the sum of squared residuals. Gradient Descent is a separate optimization approach that can minimize a cost function iteratively.