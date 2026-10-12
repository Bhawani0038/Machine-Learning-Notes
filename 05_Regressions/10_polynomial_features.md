# 10\. Polynomial Features in Scikit-learn

## 1\. What are Polynomial Features?

**Polynomial Features are new features created by raising existing input features to different powers and, when multiple inputs exist, creating their interaction terms.**

In Machine Learning, these features help a regression model learn nonlinear relationships.

For example, if the original input is\:

$$
X=[2,3,4]
$$

For degree 2, the transformed features are\:

$$
X'=[x,x^2]
$$

Therefore\:

| Original $x$ | $x^2$ |
|--------|--------|
| 2 | 4 |
| 3 | 9 |
| 4 | 16 |

The model can now learn this equation\:

$$
\hat y=b_0+b_1x+b_2x^2
$$

## 2\. Why Do We Need Polynomial Features?

Consider predicting sales based on advertising expenditure.

A straight-line model assumes a constant rate of change. In reality, sales might increase rapidly at first and then grow more slowly.

Polynomial features allow the model to learn curved relationships.

- Degree 1: straight-line relationship.
- Degree 2: can capture quadratic curvature.
- Degree 3: can capture more complex curvature.
- Higher degrees: greater flexibility, but potentially greater risk of overfitting.

## 3\. Using `PolynomialFeatures`

The class is available in Scikit-learn\:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Its important parameters are\:

| Parameter | Meaning |
|--------|--------|
| `degree` | Maximum polynomial degree |
| `include_bias` | Whether to include a column of ones |
| `interaction_only` | Whether to generate only interaction terms, excluding powers such as $x^2$ |
| `order` | Memory layout of the output array (`C` or `F`\) |

For most introductory regression workflows, `degree` and `include_bias` are the key parameters.

## 4\. Example 1 - Degree 2

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2], [3], [4]])

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)

print(X_poly)
print(poly.get_feature_names_out(["x"]))
```

Output\:

```text
[[ 2.  4.]
 [ 3.  9.]
 [ 4. 16.]]

['x' 'x^2']
```

The original column `x` remains, and the new column `x^2` is added.

## 5\. Example 2 - Degree 3

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2], [3], [4]])

poly = PolynomialFeatures(
    degree=3,
    include_bias=False
)

X_poly = poly.fit_transform(X)

print(X_poly)
print(poly.get_feature_names_out(["x"]))
```

Output\:

```text
[[ 2.  4.  8.]
 [ 3.  9. 27.]
 [ 4. 16. 64.]]

['x' 'x^2' 'x^3']
```

For degree 3, the transformed features are\:

$$
X'=[x,x^2,x^3]
$$

The regression model can learn\:

$$
\hat y=b_0+b_1x+b_2x^2+b_3x^3
$$

## 6\. What Happens with Multiple Input Features?

Suppose we have two input features\:

- $x_1$\: House area
- $x_2$\: Number of rooms

For degree 2, the transformed features include\:

$$
x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2
$$

Let's see this in Python.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([
    [2, 3],
    [4, 5]
])

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)

print(poly.get_feature_names_out(["area", "rooms"]))
print(X_poly)
```

Output\:

```text
['area' 'rooms' 'area^2' 'area rooms' 'rooms^2']

[[ 2.  3.  4.  6.  9.]
 [ 4.  5. 16. 20. 25.]]
```

Notice the interaction feature `area rooms`\. It lets the model learn whether the combined effect of area and rooms contributes to the target beyond their individual linear effects.

## 7\. Understanding `include_bias`

```python
PolynomialFeatures(degree=2, include_bias=True)
```

This adds a column of ones\:

$$
X'=[1,x,x^2]
$$

With `include_bias=False`\:

$$
X'=[x,x^2]
$$

**Best practice\:** When using `LinearRegression()`, generally use `include_bias=False` because the estimator includes an intercept by default. This avoids an unnecessary constant feature.

## 8\. Polynomial Features with Linear Regression

A complete example using a pipeline\:

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([3, 7, 13, 21, 31, 43])

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

model.fit(X, y)

prediction = model.predict([[7]])

print("Predicted value:", round(prediction[0], 2))
```

Output\:

```text
Predicted value: 57.0
```

The model learns a quadratic relationship from the training examples. The pipeline ensures that polynomial transformation is applied consistently during training and prediction.

## 9\. Important Points

- `PolynomialFeatures` creates additional input columns; it does not train a model by itself.
- `degree=2` includes squared terms and, with multiple features, pairwise interactions.
- Polynomial Regression remains linear in its learned coefficients.
- Higher polynomial degrees can increase overfitting risk and computational cost.
- Fit transformations on training data only; use a pipeline with cross-validation to avoid data leakage.