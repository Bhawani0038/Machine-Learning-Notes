## Topic 14: Data Leakage

**Objective\:** Understand data leakage, identify common types, and learn how to prevent it when building machine learning models.

## 1\. What is data leakage?

Data leakage occurs when a machine learning model receives information during training that would not legitimately be available when making real-world predictions.

As a result, the model may show excellent performance during testing but perform poorly on new, unseen data.

**Simple example\:** Suppose we want to predict whether a telecom customer will leave the company. If we use a column called `cancellation_date`, the model may indirectly learn the answer because this information is available only after cancellation.

## 2\. Types of data leakage

| Type | Meaning | Example |
|--------|--------|--------|
| Target leakage | A feature contains information about the target or its outcome | Using cancellation status to predict churn |
| Train-test contamination | Information from the test set influences training or preprocessing | Calculating imputation values using the complete dataset |
| Temporal leakage | Future information is used to predict the past | Using next month's support calls to predict today's churn |

## 3\. Example of target leakage

Suppose our target is `churn`\.

```python
import pandas as pd

data = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003", "C004"],
    "monthly_charges": [500, 1200, 800, 1500],
    "churn": [1, 0, 0, 1],
    "cancellation_processed": [1, 0, 0, 1]
})

print(data)
```

Here, `cancellation_processed` directly reveals the churn outcome in this example.

**Incorrect approach\:**

```python
X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]
```

The feature `cancellation_processed` remains in `X`, allowing the model to learn the answer from information that would not be available before churn.

**Correct approach\:**

```python
X = data.drop(
    columns=[
        "customer_id",
        "churn",
        "cancellation_processed"
    ]
)

y = data["churn"]
```

The general rule is to remove features that reveal the target or depend on information unavailable at prediction time.

## 4\. Train-test contamination during preprocessing

Consider missing values in a numerical feature.

**Incorrect approach\:**

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_imputed = imputer.fit_transform(X)

# Split after fitting the imputer
```

The imputer calculates its mean using the complete dataset, including future test rows. This allows information from the test set to influence preprocessing.

**Correct approach\:**

```python
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy="mean")

X_train_imputed = imputer.fit_transform(X_train)

X_test_imputed = imputer.transform(X_test)
```

Remember\:

- `fit_transform()` learns preprocessing parameters from the training data and transforms it.
- `transform()` applies the already-learned parameters to the test data.
- Never fit the imputer, scaler, or encoder separately on the test set.

For categorical features, use the same principle with an appropriate categorical imputer and encoder.

## 5\. Data leakage during feature selection

Feature selection can also cause leakage if it is performed before splitting the dataset.

**Incorrect order\:**

1. Select the best features using the complete dataset and target.
2. Split the selected data into training and testing sets.
3. Train and evaluate the model.

The test labels have already influenced the feature-selection process.

**Correct order\:**

1. Split the data into training and testing sets.
2. Learn which features to select using training data only.
3. Apply the selected features consistently to the test data.
4. Evaluate the model on the untouched test set.

If you use cross-validation, learned preprocessing and feature selection should be fitted separately inside each training fold. A scikit-learn `Pipeline` is a reliable way to enforce this.

## 6\. Data leakage in time-based problems

Imagine we are predicting whether a customer will churn on **October 1**\.

| Feature | Valid? | Reason |
|--------|--------|--------|
| Tenure as of October 1 | Yes | Available at prediction time |
| Support calls up to October 1 | Yes | Available at prediction time |
| Support calls during October | No, for an October 1 prediction | Future information |
| Actual cancellation date | No | Reveals the outcome |

For time-dependent problems, use a time-based split when appropriate: train on earlier observations and evaluate on later observations.

## 7\. Practical checklist

Before training a machine learning model, ask\:

- Does any feature directly reveal the target?
- Does any feature contain information recorded after the prediction date?
- Were imputers, scalers, and encoders fitted only on training data?
- Was feature selection performed using training data only?
- Were duplicate records or overlapping customers handled appropriately?
- Is the test set representative of the real prediction scenario?

**Important\:** A model achieving 100% accuracy is not automatically leaking data, but unexpectedly perfect performance is a good reason to investigate.