# 7\. Mean Squared Error (MSE)

## 1\. What is Mean Squared Error?

**Mean Squared Error (MSE) is a regression evaluation metric that calculates the average of the squared differences between actual and predicted values.**

It helps us understand how far a regression model's predictions are from the actual values.

- Lower MSE generally indicates smaller prediction errors on the evaluated dataset.
- MSE is always non-negative.
- MSE equals zero when every prediction matches its actual value.
- Large errors receive disproportionately greater penalties because errors are squared.

## 2\. Mathematical Formula

$$
\boxed{MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Where\:

- $n$ \= Number of observations
- $y_i$ \= Actual value
- $\hat{y}_i$ \= Predicted value
- $y_i-\hat{y}_i$ \= Prediction error

## 3\. Step-by-Step Calculation

Suppose we predict student marks.

| Student | Actual Marks | Predicted Marks | Error | Squared Error |
|--------|--------|--------|--------|--------|
| A | 50 | 48 | 2 | 4 |
| B | 60 | 62 | \-2 | 4 |
| C | 70 | 65 | 5 | 25 |
| D | 80 | 78 | 2 | 4 |
| **Total** |  |  |  | **37** |

Step 1: Calculate the error.

$$
e_i=y_i-\hat{y}_i
$$

Step 2: Square each error.

$$
2^2+(-2)^2+5^2+2^2=37
$$

Step 3: Divide by the number of observations.

$$
MSE=\frac{37}{4}=\boxed{9.25}
$$

The model's MSE is **9.25 marks squared**\.

## 4\. Python Implementation

### Method 1: Using Scikit-learn

```python
from sklearn.metrics import mean_squared_error

actual = [50, 60, 70, 80]
predicted = [48, 62, 65, 78]

mse = mean_squared_error(actual, predicted)

print("Mean Squared Error:", mse)
```

Output\:

```text
Mean Squared Error: 9.25
```

### Method 2: Using NumPy

```python
import numpy as np

actual = np.array([50, 60, 70, 80])
predicted = np.array([48, 62, 65, 78])

errors = actual - predicted
squared_errors = errors ** 2

mse = np.mean(squared_errors)

print("Errors:", errors)
print("Squared Errors:", squared_errors)
print("MSE:", mse)
```

Output\:

```text
Errors: [ 2 -2  5  2]
Squared Errors: [ 4  4 25  4]
MSE: 9.25
```

## 5\. MSE vs MAE vs RMSE

| Metric | Formula | Key characteristic |
|--------|--------|--------|
| MAE | $\frac{1}{n}\sum \|y_i-\hat y_i\|$ | Average absolute error |
| MSE | $\frac{1}{n}\sum (y_i-\hat y_i)^2$ | Penalizes large errors more heavily |
| RMSE | $\sqrt{MSE}$ | Expressed in the target's original units |

For the example above\:

$$
MAE=\frac{2+2+5+2}{4}=2.75
$$

$$
MSE=9.25
$$

$$
RMSE=\sqrt{9.25}\approx3.04
$$

### Compare the three metrics

MAE

### 2.75

MSE

### 9.25

RMSE

### 3.04

Bars use a common illustrative scale for display only; MSE has squared units, so its magnitude is not directly comparable to MAE or RMSE.

## 6\. Why Can MSE Be Sensitive to Outliers?

Consider two prediction errors: 2 and 10.

- Squared error for 2: $2^2=4$
- Squared error for 10: $10^2=100$

An error five times larger contributes 25 times as much squared error. Therefore, MSE can be strongly influenced by a small number of large mistakes.

For house-price prediction, MSE is useful when large prediction errors should receive extra penalties. MAE may be easier to interpret when you want the average absolute error in rupees.

## 7\. Important Points for Students

- MSE is mainly used for regression problems.
- Its value is zero or positive.
- Lower MSE is preferable when comparing models on the same evaluation dataset and target scale.
- MSE is expressed in squared target units.
- Always evaluate models on unseen test data or through appropriate cross-validation when estimating generalization performance.