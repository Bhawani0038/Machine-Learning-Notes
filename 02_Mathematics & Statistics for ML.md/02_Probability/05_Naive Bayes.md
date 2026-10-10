# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 12: Naive Bayes — Intuition

### What We Will Cover
- What is Naive Bayes?
- Why "Naive"?
- Bayes Theorem → Naive Bayes
- Classification using Probabilities
- Feature Independence Assumption
- Simple Student Example
- Spam Detection Example
- Advantages & Limitations
- Key Points

---

## 1. What is Naive Bayes?

**Naive Bayes is a supervised Machine Learning classification algorithm based on Bayes Theorem.**

It predicts the most likely class using:

- Prior probability
- Feature probabilities
- Bayes Theorem

Common applications:

- Spam detection
- Sentiment classification
- Document classification
- Text classification
- Medical classification

---

# 2. Why is it Called "Naive"?

Naive Bayes makes a **strong assumption**:

> The features are conditionally independent given the class.

For example, suppose we predict whether a student will pass using:

```text
study_hours
attendance
```

Naive Bayes approximately assumes:

```text
study_hours
     ↓
   PASS

attendance
     ↓
   PASS
```

The features are treated as independent **given the class**.

In real life, study hours and attendance may be related.

That's why the assumption is called **naive**.

---

# 3. From Bayes Theorem to Naive Bayes

Bayes Theorem:

$$
P(C \mid X)=
\frac{P(X \mid C)P(C)}
{P(X)}
$$

Suppose we have multiple features:

$$
X=(X_1,X_2,X_3)
$$

Then:

$$
P(C \mid X_1,X_2,X_3)
$$

Naive Bayes assumes conditional independence:

$$
P(X_1,X_2,X_3 \mid C)
\approx
P(X_1 \mid C)P(X_2 \mid C)P(X_3 \mid C)
$$

Therefore:

$$
P(C \mid X)
\propto
P(C)
\prod_i P(X_i \mid C)
$$

For classification, we compare the probabilities for each class and choose the largest.

---

# 4. Simple Student Example

Suppose we want to predict:

```text
PASS
FAIL
```

using:

```text
Study Hours
```

Historical data:

| Study Hours | Result |
|---:|---|
| 1 | Fail |
| 2 | Fail |
| 2 | Fail |
| 3 | Pass |
| 4 | Pass |
| 5 | Pass |
| 5 | Pass |

Suppose a new student studies for:

```text
3 hours
```

We want:

$$
P(\text{Pass} \mid 3\ \text{hours})
$$

and:

$$
P(\text{Fail} \mid 3\ \text{hours})
$$

The model compares these probabilities.

The class with the larger probability becomes the prediction.

---

# 5. Text Classification Example

Consider an email:

```text
"Win free money"
```

Naive Bayes can estimate:

$$
P(\text{Spam} \mid \text{Win},\text{Free},\text{Money})
$$

and:

$$
P(\text{Not Spam} \mid \text{Win},\text{Free},\text{Money})
$$

If:

$$
P(\text{Spam} \mid \text{words}) = 0.92
$$

and:

$$
P(\text{Not Spam} \mid \text{words}) = 0.08
$$

the prediction is:

```text
Spam
```

---

# 6. Why Naive Bayes Works Well for Text

Text datasets can contain thousands of features.

For example:

```text
free
money
offer
discount
meeting
project
invoice
hello
...
```

Naive Bayes can efficiently calculate probabilities for these features.

That's why variants such as **Multinomial Naive Bayes** are commonly used for text classification.

---

# 7. Main Naive Bayes Variants

| Variant | Typical Use |
|---|---|
| Gaussian Naive Bayes | Continuous numerical features |
| Multinomial Naive Bayes | Word counts / text |
| Bernoulli Naive Bayes | Binary features |

### Gaussian Naive Bayes

Useful when numerical features are approximately Gaussian/normal within each class.

Example:

```text
age
salary
blood_pressure
```

### Multinomial Naive Bayes

Common for:

```text
word counts
TF-IDF features
text classification
```

### Bernoulli Naive Bayes

Useful when features represent:

```text
0 = word absent
1 = word present
```

---

# 8. Advantages

- Simple to understand
- Very fast to train
- Fast prediction
- Works well with high-dimensional data
- Requires relatively little training data
- Excellent baseline for text classification
- Produces probability estimates

---

# 9. Limitations

- Independence assumption may be unrealistic.
- Correlated features can reduce performance.
- Probability estimates may require calibration depending on the application.
- Continuous-feature assumptions depend on the chosen variant.
- It may perform poorly when relationships between features are complex.

---

# 10. Naive Bayes Workflow

```text
Training Data
     ↓
Calculate Class Probabilities
     ↓
Calculate Feature Probabilities
     ↓
Apply Bayes Formula
     ↓
Compare Class Probabilities
     ↓
Select Highest Probability
     ↓
Prediction
```

---

## Key Points

- **Naive Bayes is a classification algorithm.**
- It is based on **Bayes Theorem**.
- "Naive" refers to the **conditional independence assumption**.
- It calculates the probability of each class given the features.
- The class with the highest probability is selected.
- **Gaussian NB** → numerical features.
- **Multinomial NB** → text/count data.
- **Bernoulli NB** → binary features.
- Naive Bayes is especially useful for **text and high-dimensional classification**.