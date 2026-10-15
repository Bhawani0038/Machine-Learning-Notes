## Topic 4: Multilabel Classification

### 1\. Definition

**Multilabel Classification is a supervised machine learning task in which a single observation can belong to multiple classes simultaneously.**

Unlike multiclass classification, where we select one class from several possible classes, multilabel classification can assign several labels to the same observation.

### 2\. Real-world examples

| Problem | Input | Possible labels |
|--------|--------|--------|
| Movie classification | Movie details | Action, Comedy, Drama |
| News classification | News article | Politics, Business, Technology |
| Email classification | Email content | Work, Urgent, Meeting |
| Product tagging | Product description | Electronics, Wireless, Portable |
| Medical image analysis | Medical image | Finding A, Finding B |

For example, a movie can be both Action and Comedy. It does not have to belong to only one category.

## 3\. Multiclass vs multilabel classification

**Multiclass: one class per observation**

A fruit classifier predicts exactly one category.

(Apple) (Banana) (Orange - predicted)

---

**Multilabel: multiple labels per observation**

A movie classifier can assign several genres.

(Action) (Comedy) (Drama)

The key difference is that multiclass classification typically uses mutually exclusive classes, whereas multilabel classification allows labels to overlap.

## 4\. Understanding multilabel data

Suppose we want to classify movies by their genres.

| Movie | Action | Comedy | Drama |
|--------|--------|--------|--------|
| Movie A | 1 | 1 | 0 |
| Movie B | 0 | 1 | 1 |
| Movie C | 1 | 0 | 1 |
| Movie D | 1 | 1 | 1 |
| Movie E | 0 | 0 | 1 |

Each genre becomes a separate binary target column.

- `1` means the label applies.
- `0` means the label does not apply.

Movie D has all three labels. This is valid in multilabel classification.

## 5\. Create a dataset using Pandas

```python
import pandas as pd

df = pd.DataFrame({
    "action_score": [8, 2, 6, 9, 3, 7, 4, 10, 5, 1, 8, 6],
    "comedy_score": [7, 9, 3, 8, 6, 2, 8, 7, 4, 5, 9, 6],
    "drama_score":  [2, 7, 8, 6, 9, 5, 7, 3, 8, 9, 4, 7],
    "Action":       [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    "Comedy":       [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    "Drama":        [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
})

print(df.head())
```

The first three columns are features. The last three columns are target labels.

These values are illustrative teaching data; a production model needs representative labeled examples and independent evaluation.

## 6\. Train a multilabel classification model

Scikit-learn provides `MultiOutputClassifier`, which trains a separate classifier for each target column.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Features
X = df[["action_score", "comedy_score", "drama_score"]]

# Multiple target labels
y = df[["Action", "Comedy", "Drama"]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Create multilabel model
model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
)

# Train
model.fit(X_train, y_train)

# Predict multiple labels
y_pred = model.predict(X_test)

predictions = pd.DataFrame(
    y_pred,
    columns=y.columns,
    index=y_test.index
)

print("Actual labels:")
print(y_test)

print("\nPredicted labels:")
print(predictions)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=y.columns,
    zero_division=0
))
```

`MultiOutputClassifier` fits one binary classifier per genre. The final prediction can contain any combination of zeros and ones.

## 7\. Predict labels for a new movie

```python
new_movie = pd.DataFrame({
    "action_score": [9],
    "comedy_score": [8],
    "drama_score": [3]
})

prediction = model.predict(new_movie)[0]

predicted_genres = [
    genre
    for genre, label in zip(y.columns, prediction)
    if label == 1
]

print("Predicted genres:", predicted_genres)
```

The output may contain one, multiple, or even zero predicted genres, depending on the learned classifiers.

## 8\. Important evaluation metrics

| Metric | Purpose |
|--------|--------|
| Hamming loss | Measures the fraction of individual labels predicted incorrectly |
| Exact match ratio | Measures the fraction of observations for which every label is correct |
| Micro F1-score | Evaluates overall label-level precision and recall |
| Macro F1-score | Calculates F1 separately for each label and averages the results |

Exact match is strict: if a movie has three actual genres and the model predicts two correctly, that observation is still counted as an incorrect exact match.

## 9\. When should you use each type?

| Requirement | Classification type |
|--------|--------|
| Predict Pass or Fail | Binary |
| Predict one of five product categories | Multiclass |
| Assign multiple genres to a movie | Multilabel |
