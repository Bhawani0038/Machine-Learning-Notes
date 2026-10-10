# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 10: Conditional Probability

### What We Will Cover
- What is Conditional Probability?
- Meaning of $P(A \mid B)$
- Formula
- Simple Examples
- Dependent Events
- Real-Life ML Example
- Difference: $P(A \mid B)$ vs $P(B \mid A)$
- Python Example
- Key Points

---

## 1. What is Conditional Probability?

**Conditional probability is the probability of an event happening given that another event has already happened.**

It is written as:

$$
P(A \mid B)
$$

Read as:

> **Probability of A given B**

The vertical bar `|` means **given that**.

---

## 2. Simple Example

Suppose a class has:

- 60 students
- 40 students study regularly
- 32 regular students passed

Question:

> What is the probability that a student passed **given that the student studies regularly?**

We want:

$$
P(\text{Pass} \mid \text{Regular})
$$

Since we already know that the student is from the **regular group**, we only consider the 40 regular students.

Among them, 32 passed.

Therefore:

$$
P(\text{Pass} \mid \text{Regular}) = \frac{32}{40} = 0.8
$$

So:

**Probability = 80%**

---

# 3. Conditional Probability Formula

The general formula is:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

Where:

- $P(A \mid B)$ = probability of A given B
- $P(A \cap B)$ = probability that A and B both occur
- $P(B)$ = probability of B

### Important

The condition is the **denominator**.

For:

$$
P(A \mid B)
$$

we divide by:

$$
P(B)
$$

---

# 4. Example with Numbers

Suppose 100 students:

| | Pass | Fail | Total |
|---|---:|---:|---:|
| Regular | 32 | 8 | 40 |
| Irregular | 18 | 42 | 60 |
| **Total** | **50** | **50** | **100** |

Question:

> Probability that a student passed given that they are regular?

We need:

$$
P(\text{Pass} \mid \text{Regular})
$$

Among regular students:

```text
Total regular = 40
Regular + Pass = 32
```

Therefore:

$$
P(\text{Pass} \mid \text{Regular}) = \frac{32}{40} = 0.8
$$

**Answer = 80%**

---

# 5. $P(A \mid B)$ vs $P(B \mid A)$

These are **not generally the same**.

### Example

Suppose:

$$
P(\text{Pass} \mid \text{Regular}) = 0.8
$$

This means:

> Among regular students, 80% passed.

But:

$$
P(\text{Regular} \mid \text{Pass})
$$

means:

> Among students who passed, what percentage were regular?

From our table:

```text
Passed students = 50
Regular + Passed = 32
```

Therefore:

$$
P(\text{Regular} \mid \text{Pass}) = \frac{32}{50} = 0.64
$$

So:

$$
P(\text{Pass} \mid \text{Regular}) = 0.80
$$

but:

$$
P(\text{Regular} \mid \text{Pass}) = 0.64
$$

### Therefore:

$$
\boxed{P(A \mid B) \neq P(B \mid A)}
$$

in general.

This distinction is **extremely important for Bayes Theorem**.

---

# 6. Connection with Dependent Events

For dependent events:

$$
P(A \cap B) = P(A) \times P(B \mid A)
$$

Example:

```text
First event → A
        ↓
Changes probability of B
        ↓
P(B|A)
```

So conditional probability helps us calculate the probability of **both events occurring**.

---

# 7. Medical Example

Suppose:

- 5% of people have a disease.
- A person has tested positive.

We might ask:

> What is the probability that the person actually has the disease given that the test is positive?

This is:

$$
P(\text{Disease} \mid \text{Positive})
$$

Notice the direction:

```text
Disease given Positive
```

not:

```text
Positive given Disease
```

These are different probabilities.

This type of problem is exactly where **Bayes Theorem** becomes important.

---

# 8. Machine Learning Connection

Conditional probability is fundamental to many ML algorithms.

### Naive Bayes

Naive Bayes uses probabilities such as:

$$
P(\text{Class} \mid \text{Features})
$$

For example:

$$
P(\text{Spam} \mid \text{Words})
$$

Meaning:

> Probability that an email is spam given the words present in the email.

### Other applications

- Disease prediction
- Spam detection
- Fraud detection
- Customer churn
- Credit risk
- Recommendation systems

---

# 9. Python Example

```python
passed_regular = 32
total_regular = 40

probability = passed_regular / total_regular

print(probability)
```

Output:

```text
0.8
```

Percentage:

```python
print(probability * 100)
```

Output:

```text
80.0
```

---

## Key Points

- Conditional probability means **probability of A given B**.
- Notation:

$$
P(A \mid B)
$$

- Formula:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

- The event after `|` is the **condition**.
- $P(A \mid B)$ and $P(B \mid A)$ are generally different.
- Conditional probability is closely related to **dependent events**.
- It is a fundamental concept behind **Bayes Theorem and Naive Bayes**.