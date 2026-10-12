## Topic 15: Handling Imbalanced Data

**Objective\:** Understand imbalanced datasets and learn techniques to handle them in classification problems.

## 1\. What is imbalanced data?

A dataset is imbalanced when one target class contains significantly more observations than another class.

For example, in Customer Churn Prediction, most customers may stay with the company while only a small percentage leave.

| Customer status | Number of customers | Percentage |
|--------|--------|--------|
| No Churn | 900 | 90% |
| Churn | 100 | 10% |
| **Total** | **1,000** | **100%** |

Here, `No Churn` is the majority class, and `Churn` is the minority class.

## 2\. Why is imbalanced data a problem?

Suppose a model predicts `No Churn` for every customer.

- Correct predictions: 900
- Incorrect predictions: 100
- Accuracy: 90%

The model achieves 90% accuracy but fails to identify even one customer who will churn.

Therefore, **accuracy alone can be misleading when classes are imbalanced.**

## 3\. Check class imbalance using pandas

Using our Customer Churn dataset\:

```python
import pandas as pd

print(clean_df["churn"].value_counts())

print(clean_df["churn"].value_counts(normalize=True) * 100)
```

The first statement displays class counts. The second displays class percentages.

You can also visualize the distribution\:

```python
import matplotlib.pyplot as plt

clean_df["churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.show()
```

## 4\. Techniques to handle imbalanced data

**A. Random Under-Sampling**

Reduces the number of observations in the majority class.

Example: Reduce 900 `No Churn` observations to 100.

- Advantage: Smaller, faster training dataset.
- Disadvantage: Useful majority-class information may be lost.

**B. Random Over-Sampling**

Randomly duplicates observations from the minority class.

Example: Increase 100 `Churn` observations to 900.

- Advantage: Keeps the original majority-class observations.
- Disadvantage: Duplicating rows may cause overfitting.

**C. SMOTE**

The Synthetic Minority Over-sampling Technique generates new minority-class observations by interpolating between existing minority-class examples.

- Advantage: Creates synthetic examples instead of simply duplicating rows.
- Disadvantage: Can create unrealistic or overlapping examples, especially when classes overlap or contain noise.

## 5\. Random over-sampling using Python

First, separate the features and target, then split the data. Apply over-sampling to the **training data only**\.

```python
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

X = clean_df.drop(
    columns=["customer_id", "churn", "churn_encoded"],
    errors="ignore"
)

y = clean_df["churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_data = X_train.copy()
train_data["churn"] = y_train

majority = train_data[train_data["churn"] == 0]
minority = train_data[train_data["churn"] == 1]

minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_train = pd.concat(
    [majority, minority_oversampled],
    ignore_index=True
)

print(balanced_train["churn"].value_counts())
```

**Important\:** This example assumes the features are already suitable for resampling. If missing values remain, impute them first using parameters learned from the training set. Fit all other learned preprocessing on training data as well.

The test set remains unchanged, so evaluation reflects the original class distribution.

## 7\. Which evaluation metrics should we use?

| Metric | Why it helps |
|--------|--------|
| Precision | Measures how many predicted churners actually churned |
| Recall | Measures how many actual churners were identified |
| F1-score | Balances precision and recall |
| PR-AUC | Useful for evaluating minority-class detection, especially when positive cases are rare |
| Confusion matrix | Shows true positives, false positives, true negatives, and false negatives |

The right metric depends on business costs. For example, a telecom company may prioritize recall if missing a customer likely to churn is particularly expensive.

## 8\. Important rules

- Split the data before resampling.
- Apply under-sampling, over-sampling, or SMOTE only to training data.
- Never balance the test set just to improve reported performance.
- Compare results against a baseline model.
- Consider class weights as an alternative to resampling.
- Do not assume a 50:50 class distribution is always best.