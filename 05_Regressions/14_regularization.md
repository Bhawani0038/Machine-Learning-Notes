# 14\. Regularization in Regression

## 1\. What is Regularization?

**Regularization is a technique used to reduce overfitting by penalizing unnecessarily large model coefficients.**

In regression, a model may fit training data very closely but perform poorly on unseen data. Regularization helps control model complexity.

### Why do we need regularization?

- To reduce overfitting.
- To control large coefficients.
- To improve generalization to unseen data.
- To make models more stable when features are correlated.
- To perform feature selection with methods such as Lasso.

## 2\. Types of Regularization

| Method | Penalty | Main characteristic |
|--------|--------|--------|
| Ridge Regression | L2 | Shrinks coefficients toward zero |
| Lasso Regression | L1 | Can make coefficients exactly zero |
| Elastic Net | L1 + L2 | Combines shrinkage and feature selection |

### Mathematical formulas

Ridge\:

$$
J = SSE + \alpha\sum_{j=1}^{p}\beta_j^2
$$

Lasso\:

$$
J = SSE + \alpha\sum_{j=1}^{p}|\beta_j|
$$

Elastic Net\:

$$
J = SSE + \alpha\rho\sum_{j=1}^{p}|\beta_j|
+\frac{\alpha(1-\rho)}{2}\sum_{j=1}^{p}\beta_j^2
$$

Here, $SSE$ is the sum of squared errors, $\alpha$ controls regularization strength, and $\rho$ controls the balance between L1 and L2 penalties.

## 3\. Understanding Alpha

| Alpha | Typical effect |
|--------|--------|
| Very small | Little regularization |
| Moderate | More coefficient shrinkage |
| Very large | Greater risk of underfitting |

The best value depends on the dataset. Use cross-validation to select it.

## 4\. Python Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
    "Elastic Net": ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=10000
    )
}

for name, estimator in models.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", estimator)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(name)
    print("Test R2:", model.score(X_test, y_test))
    print()
```

This code assumes `X_train`, `X_test`, `y_train`, and `y_test` have already been created. The scaling is fitted only on the training data because it is inside the pipeline.

**Important\:** Regularization does not guarantee better performance. Compare training and test results, and tune hyperparameters using cross-validation.

---