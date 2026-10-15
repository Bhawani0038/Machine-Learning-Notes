## Topic 8: Naive Bayes

### 1\. Definition

**Naive Bayes is a supervised classification algorithm based on Bayes' theorem.** It predicts a class by calculating the probability of that class given the observed features.

It is widely used for\:

- Spam email detection
- Sentiment analysis
- News categorization
- Document classification
- Customer feedback classification

It is called *naive* because it assumes that the features are conditionally independent given the class. This assumption is often unrealistic, but the algorithm can still perform well on many practical problems.

## 2\. Understanding the intuition

Suppose an email contains the words\:

`Win money now`

We want to determine whether the email is Spam or Not Spam.

A Naive Bayes model learns patterns from previously labeled emails. If these words are more strongly associated with spam messages in the training data, the model may predict Spam.

It considers both\:

- The prior probability of a class, such as how common spam is.
- The likelihood of observing the words given that class.

## 3\. Bayes' theorem

The fundamental equation is\:

$$
\boxed{P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}}
$$

Where\:

| Symbol | Meaning |
|--------|--------|
| $P(C\mid X)$ | Probability of class $C$ given the observed features |
| $P(X\mid C)$ | Likelihood of the features given class $C$ |
| $P(C)$ | Prior probability of class $C$ |
| $P(X)$ | Overall probability of the observed features |

For classification, $P(X)$ is the same across candidate classes for a given observation. Therefore, we can compare the unnormalized scores\:

$$
P(C\mid X)\propto P(X\mid C)P(C)
$$

The model predicts the class with the largest posterior probability.

## 4\. Why is it called Naive Bayes?

Suppose the email contains three features\:

- Word `win`
- Word `money`
- Word `offer`

Naive Bayes makes the conditional-independence approximation\:

$$
P(X\mid C)\approx
P(\text{win}\mid C)
P(\text{money}\mid C)
P(\text{offer}\mid C)
$$

This lets us estimate the combined likelihood from individual feature likelihoods instead of estimating every possible combination.

For text classification, this approximation is useful because documents can contain thousands of possible words.

## 5\. Mathematical example

Suppose an email classifier has learned these probabilities\:

$$
P(\text{Spam})=0.4
$$

$$
P(\text{Not Spam})=0.6
$$

For an email containing `win` and `money`, suppose the learned word likelihoods are\:

| Probability | Spam | Not Spam |
|--------|--------|--------|
| Prior probability | 0.40 | 0.60 |
| $P(\text{win}\mid C)$ | 0.70 | 0.10 |
| $P(\text{money}\mid C)$ | 0.60 | 0.20 |

**Step 1: Calculate the Spam score.**

$$
0.40\times0.70\times0.60=0.168
$$

**Step 2: Calculate the Not Spam score.**

$$
0.60\times0.10\times0.20=0.012
$$

**Step 3: Normalize the scores.**

$$
P(\text{Spam}\mid X)
=\frac{0.168}{0.168+0.012}
\approx0.9333
$$

$$
P(\text{Not Spam}\mid X)\approx0.0667
$$

The predicted class is **Spam**, with an estimated posterior probability of approximately 93.33% under these assumed likelihoods.

These values are illustrative, not probabilities measured from a real email dataset.

## 6\. Practical Python implementation

We will use `CountVectorizer` to convert text into numerical features and `MultinomialNB` to classify emails.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.DataFrame({
    "message": [
        "Win money now",
        "Claim your free prize",
        "Limited offer win today",
        "Free cash reward",
        "Win a free gift",
        "Meeting scheduled for tomorrow",
        "Please review the project report",
        "Lunch meeting at noon",
        "Project deadline is Friday",
        "Can we discuss the report",
        "Please send the meeting notes",
        "The project review is complete"
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "not_spam", "not_spam", "not_spam",
        "not_spam", "not_spam", "not_spam", "not_spam"
    ]
})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB(alpha=1.0)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

new_emails = [
    "Win free money today",
    "Please review the project report"
]

print(model.predict(new_emails))
```

The training dataset is intentionally small for demonstration. A production spam filter needs substantially more representative emails, independent evaluation, and monitoring for changing spam patterns.

## 7\. Types of Naive Bayes

| Algorithm | Suitable data | Common application |
|--------|--------|--------|
| Gaussian Naive Bayes | Continuous numerical features | Basic numerical classification |
| Multinomial Naive Bayes | Non-negative counts or frequencies | Text classification |
| Bernoulli Naive Bayes | Binary features | Word presence or absence |

Choose the variant based on how the features are represented and the assumptions appropriate for the data.

## 8\. Advantages and limitations

**Advantages**

- Fast to train and predict.
- Works well with high-dimensional text data.
- Supports multiclass classification.
- Requires relatively little training data for some tasks.

**Limitations**

- The conditional-independence assumption may be unrealistic.
- Correlated features can reduce the usefulness of its probability estimates.
- Performance depends on the data representation and smoothing settings.
- Predicted probabilities may be poorly calibrated.
