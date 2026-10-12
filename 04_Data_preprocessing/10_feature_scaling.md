## Topic 10: Feature Scaling

**Objective\:** Learn how to bring numerical features onto comparable scales using Standardization and Normalization, and understand which machine learning algorithms benefit from scaling.

We will use the same Customer Churn dataset with features such as `age`, `monthly_charges`, `tenure_months`, and `support_calls`\.

## 1\. Why do we need feature scaling?

Our numerical features have very different ranges.

| Feature | Example range |
|--------|--------|
| `age` | 18-100 |
| `monthly_charges` | ₹500-₹50,000 |
| `tenure_months` | 0-120 |
| `support_calls` | 0-20 |

Some algorithms use distances, gradients, or regularization. Features with larger numerical scales can disproportionately influence these calculations.

Feature scaling changes the numerical scale of features without intentionally changing their underlying meaning.

**Algorithms that commonly benefit from scaling\:**

- K-Nearest Neighbors (KNN)
- K-Means clustering
- Support Vector Machines (SVM)
- Logistic Regression with regularization
- Linear Regression with regularization, such as Ridge and Lasso
- Neural Networks

Decision Trees and Random Forests generally do not require feature scaling.

## 2\. Standardization

Standardization transforms a feature by subtracting its mean and dividing by its standard deviation.

$$
z=\frac{x-\mu}{\sigma}
$$

Where\:

- $x$ \= original value
- $\mu$ \= feature mean
- $\sigma$ \= feature standard deviation
- $z$ \= standardized value

After standardization, a feature generally has mean close to 0 and standard deviation close to 1 when evaluated on the data used to fit the scaler.

### A. Apply StandardScaler

```python
from sklearn.preprocessing import StandardScaler

numerical_columns = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls"
]

scaler = StandardScaler()

scaled_array = scaler.fit_transform(
    clean_df[numerical_columns]
)

scaled_df = pd.DataFrame(
    scaled_array,
    columns=numerical_columns,
    index=clean_df.index
)

print(scaled_df.head())
```

`StandardScaler` learns the mean and standard deviation of each feature, then applies the transformation.

### B. Understand the result

Suppose a feature has a mean of 30 and a standard deviation of 10.

For an observation with value 40\:

$$
z=\frac{40-30}{10}=1
$$

The standardized value is `1`, meaning the observation is one standard deviation above the mean.

Standardization does **not** guarantee that every value falls between -1 and 1. Extreme values can still produce large standardized values.

## 3\. Normalization using MinMaxScaler

In many machine learning contexts, normalization refers to rescaling each feature to a specified range, commonly 0 to 1.

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

The minimum becomes 0 and the maximum becomes 1 for a non-constant feature when the scaler is fitted on that data.

### Apply MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

normalized_array = minmax_scaler.fit_transform(
    clean_df[numerical_columns]
)

normalized_df = pd.DataFrame(
    normalized_array,
    columns=numerical_columns,
    index=clean_df.index
)

print(normalized_df.head())
```

### Example

Suppose monthly charges range from ₹500 to ₹2,500.

For a charge of ₹1,500\:

$$
x'=\frac{1500-500}{2500-500}=0.5
$$

The normalized value is `0.5`\.

**Limitation\:** Min-Max scaling is sensitive to outliers. If the maximum charge is ₹50,000, ordinary charges may be compressed into a small part of the 0-1 range.

## 4\. Compare Standardization and Normalization

| Property | StandardScaler | MinMaxScaler |
|--------|--------|--------|
| Main purpose | Center and scale features | Map features to a fixed range |
| Formula | $(x-\mu)/\sigma$ | $(x-x_{\min})/(x_{\max}-x_{\min})$ |
| Typical output | Mean 0, standard deviation 1 on fitted data | Range 0-1 on fitted data |
| Sensitive to outliers | Yes | Yes, often strongly |
| Preserves original units | No | No |
| Common use | Logistic Regression, SVM, regularized models | Neural networks and algorithms needing bounded inputs |

Neither method is universally superior. The best choice depends on the data, algorithm, and evaluation results.

## 5\. Critical rule: Fit the scaler on training data only

Do not scale the complete dataset before splitting it. The scaler would learn information about the test-set distribution.

Correct approach\:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = clean_df[
    ["age", "monthly_charges", "tenure_months", "support_calls"]
]

y = clean_df["churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- `fit_transform()` learns scaling parameters from the training data and transforms it.
- `transform()` applies those same parameters to the test data.

In later topics, we will automate this safely using `Pipeline` and `ColumnTransformer`\.