# 8\. Gradient Descent

## 1\. What is Gradient Descent?

**Gradient Descent is an optimization algorithm that minimizes a cost function by repeatedly updating a model's parameters in the direction that reduces the cost.**

In Machine Learning, a model may have parameters that need to be optimized to make better predictions.

For Linear Regression, these parameters are\:

- **Slope ($m$\)** - controls how much the prediction changes with the input.
- **Intercept ($c$\)** - controls the baseline prediction.

Gradient Descent adjusts these parameters iteratively to reduce prediction error.

## 2\. Real-Life Example

Suppose we want to predict student marks using study hours.

$$
\hat y=mx+c
$$

Initially, the model may use\:

$$
\hat y=2x+5
$$

But the actual data may follow a relationship closer to\:

$$
y=10x+10
$$

The initial model makes inaccurate predictions. Gradient Descent can iteratively update $m$ and $c$ to reduce the cost.

### How Gradient Descent works

**1\. Initialize parameters**

Choose initial slope and intercept

**2\. Calculate predictions and cost**

Measure how far predictions are from actual values

**3\. Calculate gradients**

Determine which direction reduces the cost

**4\. Update parameters**

Repeat until a stopping condition is reached

## 3\. Mathematical Formula

The parameter update rule is\:

$$
\boxed{\theta_{\text{new}}=\theta_{\text{old}}-\alpha\frac{\partial J}{\partial\theta}}
$$

Where\:

- $\theta$ \= Model parameter, such as slope or intercept
- $J$ \= Cost function
- $\frac{\partial J}{\partial\theta}$ \= Gradient of the cost with respect to the parameter
- $\alpha$ \= Learning rate

The gradient indicates how the cost changes when a parameter changes slightly.

**Why subtract the gradient?** The negative gradient points in the direction of steepest local decrease in the cost function.

### Gradient Descent for Linear Regression

Using Mean Squared Error\:

$$
J(m,c)=\frac{1}{n}\sum_{i=1}^{n}(mx_i+c-y_i)^2
$$

The gradients are\:

$$
\frac{\partial J}{\partial m}
=\frac{2}{n}\sum_{i=1}^{n}x_i(mx_i+c-y_i)
$$

$$
\frac{\partial J}{\partial c}
=\frac{2}{n}\sum_{i=1}^{n}(mx_i+c-y_i)
$$

Update both parameters\:

$$
m\leftarrow m-\alpha\frac{\partial J}{\partial m}
$$

$$
c\leftarrow c-\alpha\frac{\partial J}{\partial c}
$$

These updates are performed repeatedly using the training data.

## 4\. Understanding the Learning Rate

The learning rate controls the size of each parameter update.

| Learning Rate | Possible Effect |
|--------|--------|
| Too small | Convergence may be very slow |
| Appropriate | Cost generally decreases steadily |
| Too large | Cost may oscillate or diverge |

For example, if the gradient is $8$ and the learning rate is $0.1$\:

$$
\theta_{\text{new}}=\theta_{\text{old}}-0.1(8)
$$

$$
\theta_{\text{new}}=\theta_{\text{old}}-0.8
$$

The parameter decreases by $0.8$\.

## 5\. Python Implementation From Scratch

The following example implements Batch Gradient Descent for a simple linear model without using a prebuilt regression estimator.

```python
import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([20, 30, 40, 50, 60], dtype=float)

# Initial parameters
m = 0.0
c = 0.0

learning_rate = 0.01
epochs = 2000
n = len(X)

for epoch in range(epochs):
    # Predictions
    y_pred = m * X + c

    # Gradients of MSE
    dm = (2 / n) * np.sum(X * (y_pred - y))
    dc = (2 / n) * np.sum(y_pred - y)

    # Update parameters
    m -= learning_rate * dm
    c -= learning_rate * dc

# Final predictions and cost
y_pred = m * X + c
mse = np.mean((y - y_pred) ** 2)

print("Slope:", round(m, 2))
print("Intercept:", round(c, 2))
print("MSE:", round(mse, 4))
print("Prediction for 6 hours:", round(m * 6 + c, 2))
```

The learned parameters should approach the underlying relationship $y=10x+10$, although the exact result depends on the learning rate, iteration count and numerical convergence.

```python

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Training data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([20, 30, 40, 50, 60])

# Create model using gradient descent
model = SGDRegressor(
    loss="squared_error",
    learning_rate="constant",
    eta0=0.01,
    max_iter=2000,
    tol=None,
    random_state=42
)

# Train the model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Model parameters
m = model.coef_[0]
c = model.intercept_[0]

# Evaluation
mse = mean_squared_error(y, y_pred)

print("Slope:", round(m, 2))
print("Intercept:", round(c, 2))
print("MSE:", round(mse, 4))
print("Prediction for 6 hours:", round(model.predict([[6]])[0], 2))

```

### Code explanation

- `m` and `c` start at zero.
- `y_pred` calculates the current predictions.
- `dm` and `dc` calculate the gradients.
- The update statements move the parameters in the negative-gradient direction.
- `epochs` determines the maximum number of update iterations.
- The final MSE measures the remaining error on the training dataset.

## 6\. Gradient Descent vs Ordinary Linear Regression

| Ordinary Least Squares | Gradient Descent |
|--------|--------|
| Finds coefficients by solving a least-squares optimization problem | Finds parameters through iterative updates |
| Scikit-learn's `LinearRegression` uses least-squares solvers | Can be implemented manually or with estimators such as `SGDRegressor` |
| Does not require choosing a learning rate | Requires a suitable learning rate |
| Often effective for small and medium-sized datasets | Can be useful for large datasets and iterative optimization |

**Important\:** Gradient Descent does not guarantee a good solution just because it runs for many iterations. Learning rate, feature scaling, convergence criteria and the cost function all matter.

---