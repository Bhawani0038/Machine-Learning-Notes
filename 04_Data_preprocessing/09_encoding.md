## Topic 9: Encoding

**Objective\:** Convert categorical data into numerical representations that machine learning algorithms can use, while preserving the meaning of the original features.

We will use the same Customer Churn dataset and cover two techniques\:

1. Label Encoding
2. One-Hot Encoding

## 1\. Why do we need encoding?

Consider these records\:

| gender | contract\_type | payment\_method | churn |
|--------|--------|--------|--------|
| Male | Monthly | UPI | Yes |
| Female | Yearly | Card | No |
| Male | Two Year | Bank Transfer | No |

Most traditional scikit-learn estimators expect numerical input features. We therefore need to represent categorical values numerically.

However, assigning arbitrary numbers to categories can accidentally introduce an ordering that does not exist.

---

## 2\. Label Encoding

Label Encoding assigns an integer to each distinct category.

For example\:

| Original value | Encoded value |
|--------|--------|
| No | 0 |
| Yes | 1 |

### A. Encode the target variable

For our classification problem, `churn` is the target. We can use `LabelEncoder` to convert its labels into integers.

```python
from sklearn.preprocessing import LabelEncoder

target_encoder = LabelEncoder()

clean_df["churn_encoded"] = target_encoder.fit_transform(
    clean_df["churn"]
)

print(clean_df[["churn", "churn_encoded"]].drop_duplicates())
```

Expected mapping\:

```text
   churn  churn_encoded
0    Yes              1
1     No              0
```

`LabelEncoder` sorts the class labels, so `No` becomes `0` and `Yes` becomes `1`\.

To retrieve the learned mapping\:

```python
print(target_encoder.classes_)
```

Output\:

```text
['No' 'Yes']
```

**Important\:** Keep the original target column if you need readable labels later. In a production workflow, split the dataset first and fit target encoders on training labels, or use an appropriate target-aware workflow.

### B. Why not use Label Encoding for every input feature?

Suppose we encode payment methods like this\:

| Payment method | Encoded value |
|--------|--------|
| Bank Transfer | 0 |
| Card | 1 |
| UPI | 2 |

A model may interpret the values as though UPI is greater than Card and Card is greater than Bank Transfer. These categories have no such natural ordering.

For nominal input features, One-Hot Encoding is usually more appropriate.

---

## 3\. One-Hot Encoding

One-Hot Encoding creates a separate binary column for each category.

Suppose the original feature is `payment_method`\.

| payment\_method | Bank Transfer | Card | UPI |
|--------|--------|--------|--------|
| UPI | 0 | 0 | 1 |
| Card | 0 | 1 | 0 |
| Bank Transfer | 1 | 0 | 0 |

Each row has a `1` in the column corresponding to its category and `0` in the others.

### A. Using pandas

```python
encoded_df = pd.get_dummies(
    clean_df,
    columns=["payment_method"],
    dtype=int
)

print(encoded_df.head())
```

This is useful for learning and exploratory analysis.

### B. Using scikit-learn

Scikit-learn's `OneHotEncoder` is especially useful in machine learning pipelines.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

payment_encoded = encoder.fit_transform(
    clean_df[["payment_method"]]
)

encoded_columns = encoder.get_feature_names_out(
    ["payment_method"]
)

payment_encoded_df = pd.DataFrame(
    payment_encoded,
    columns=encoded_columns,
    index=clean_df.index
)

print(payment_encoded_df.head())
```

`handle_unknown="ignore"` prevents an error when the encoder encounters a previously unseen category during transformation. Such a category produces zeros across the encoded columns for that feature.

` sparse_output=False` returns a dense array. For large datasets with many categories, the default sparse output is generally more memory-efficient.

---

## 4\. Which technique should we use?

| Situation | Recommended approach |
|--------|--------|
| Binary target such as `churn` | Label encoding or explicit binary mapping |
| Nominal feature such as `gender` | One-Hot Encoding |
| Nominal feature such as `payment_method` | One-Hot Encoding |
| Ordered categories such as Low, Medium, High | Ordinal Encoding |
| High-cardinality feature such as thousands of city names | Consider alternatives after evaluating cardinality and model requirements |

For `contract_type`, do not automatically treat Monthly, Yearly, and Two Year as ordinal categories. The correct representation depends on whether you want to encode contract duration as an ordered business concept or represent each contract type as a distinct nominal category.