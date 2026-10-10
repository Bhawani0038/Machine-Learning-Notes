# Module 2 — Mathematics & Statistics for Machine Learning
## Topic 6: Covariance

### What We Will Cover

- What is Covariance?
- Positive Covariance
- Negative Covariance
- Zero Covariance
- Covariance Formula
- Simple Example
- Covariance vs Variance
- Covariance vs Correlation
- Real-Life ML Examples
- Key Points

---

# 1. What is Covariance?

**Covariance measures how two variables change together.**

It tells us whether two variables tend to:

- Increase together → **Positive covariance**
- Move in opposite directions → **Negative covariance**
- Have little/no linear relationship → **Covariance near zero**

### Simple Intuition

Suppose:

```text
Study Hours ↑
       ↓
Marks ↑
```

As study hours increase, marks also tend to increase.

Therefore:

**Positive covariance**

---

# 2. Positive Covariance

Consider:

| Study Hours | Marks |
|---:|---:|
| 1 | 40 |
| 2 | 50 |
| 3 | 60 |
| 4 | 70 |
| 5 | 80 |

Both variables increase together.

```text
Study Hours ↑
      ↘
       Marks ↑
```

So covariance is **positive**.

---

# 3. Negative Covariance

Consider:

| Exercise Hours | Weight |
|---:|---:|
| 1 | 80 |
| 2 | 77 |
| 3 | 74 |
| 4 | 71 |
| 5 | 68 |

As exercise increases, weight tends to decrease.

Therefore:

**Negative covariance**

```text
Exercise ↑
     ↓
Weight ↓
```

---

# 4. Zero or Near-Zero Covariance

Suppose:

| Hours Studied | Shoe Size |
|---:|---:|
| 1 | 8 |
| 2 | 9 |
| 3 | 8 |
| 4 | 10 |
| 5 | 9 |

There is no clear tendency for shoe size to increase or decrease as study hours increase.

Covariance may be close to:

```text
0
```

---

# 5. Covariance Formula

For a population:

$$
Cov(X,Y)=
\frac{\sum_{i=1}^{N}(x_i-\mu_X)(y_i-\mu_Y)}
{N}
$$

Where:

- `X` = First variable
- `Y` = Second variable
- `μX` = Mean of X
- `μY` = Mean of Y
- `N` = Number of observations

The important part is:

$$
(x_i-\mu_X)(y_i-\mu_Y)
$$

We compare how each variable deviates from its mean.

---

# 6. Simple Example

Consider:

```text
X = [1, 2, 3]
Y = [2, 4, 6]
```

### Step 1 — Calculate Means

$$
Mean(X)=2
$$

$$
Mean(Y)=4
$$

### Step 2 — Calculate Deviations

| X | X − Mean X | Y | Y − Mean Y |
|---:|---:|---:|---:|
| 1 | -1 | 2 | -2 |
| 2 | 0 | 4 | 0 |
| 3 | +1 | 6 | +2 |

### Step 3 — Multiply Deviations

| X deviation | Y deviation | Product |
|---:|---:|---:|
| -1 | -2 | +2 |
| 0 | 0 | 0 |
| +1 | +2 | +2 |

The products are mostly positive.

Therefore, covariance is positive.

For the population:

$$
Cov(X,Y)=\frac{2+0+2}{3}
$$

$$
Cov(X,Y)=1.33
$$

---

# 7. Why Does the Sign Matter?

The **sign** of covariance is the most important basic interpretation.

| Covariance | Meaning |
|---|---|
| Positive | Variables tend to move together |
| Negative | Variables tend to move in opposite directions |
| Near 0 | Little linear co-movement |

Remember:

> **Covariance tells us the direction of the relationship, not a standardized strength.**

---

# 8. Covariance in Python

Using NumPy:

```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

covariance = np.cov(x, y, ddof=0)[0, 1]

print(covariance)
```

Output:

```text
1.3333333333333333
```

`ddof=0` gives the population covariance.

For sample covariance, use:

```python
np.cov(x, y, ddof=1)[0, 1]
```

---

# 9. Covariance vs Variance

Variance measures how **one variable** changes around its mean.

Covariance measures how **two variables** change together.

| Variance | Covariance |
|---|---|
| One variable | Two variables |
| `Var(X)` | `Cov(X,Y)` |
| Measures spread | Measures joint movement |
| Always ≥ 0 | Can be positive, negative, or near 0 |

Interestingly:

$$
Cov(X,X)=Var(X)
$$

---

# 10. Covariance vs Correlation

These concepts are related but different.

### Covariance

Tells us the **direction** of relationship.

```text
Positive → move together
Negative → move oppositely
```

But its magnitude depends on the units of the variables.

### Correlation

Correlation is a **standardized measure** of linear relationship.

It ranges from:

$$
-1 \le r \le 1
$$

So:

```text
+1 → Perfect positive linear relationship
 0 → No linear correlation
-1 → Perfect negative linear relationship
```

We will study correlation in the next topic.

---

# 11. Real-World ML Example

Suppose we have:

```text
sqft_area
price
```

Generally:

```text
sqft_area ↑
     ↓
price ↑
```

So we may observe **positive covariance**.

Another example:

```text
distance_from_city_center ↑
        ↓
house_price ↓
```

If this pattern exists in the data, covariance may be negative.

---

# 12. Important Limitation

Covariance near zero does **not necessarily mean there is no relationship at all**.

For example, two variables may have a strong **non-linear** relationship but covariance close to zero.

So covariance mainly captures **linear co-movement**.

This is one reason why we also study:

- Correlation
- Scatter plots
- Other statistical methods

---

## Key Points

- **Covariance measures how two variables change together.**
- Positive covariance → variables tend to move in the same direction.
- Negative covariance → variables tend to move in opposite directions.
- Covariance near zero → little linear co-movement.
- Variance deals with **one variable**.
- Covariance deals with **two variables**.
- Covariance depends on the units of the variables.
- Correlation is a standardized form of linear association and ranges from **−1 to +1**.
- Covariance is useful for understanding relationships between features before building ML models.