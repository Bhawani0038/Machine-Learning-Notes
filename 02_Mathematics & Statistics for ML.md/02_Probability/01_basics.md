# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 8: Probability — Basic Concepts

### What We Will Cover
- What is Probability?
- Experiment
- Outcome
- Sample Space
- Event
- Probability Formula
- Probability Range
- Complement
- Simple Examples
- ML Applications
- Python Implementation
- Key Points

---

## 1. What is Probability?

**Probability measures how likely an event is to happen.**

Its value is always between:

$$
0 \leq P(A) \leq 1
$$

Where:

- `0` → impossible
- `1` → certain
- `0.5` → 50% chance

Example:

> What is the probability of getting Head when tossing a fair coin?

There are two possible outcomes:

```text
Head, Tail
```

Therefore:

$$
P(\text{Head}) = \frac{1}{2} = 0.5
$$

---

## 2. Experiment

An **experiment** is an action or process that produces an outcome.

Examples:

- Tossing a coin
- Rolling a dice
- Drawing a card
- Selecting a student
- Testing a patient
- Predicting whether a customer will churn

---

## 3. Outcome

An **outcome** is a possible result of an experiment.

### Coin

```text
Head
Tail
```

### Dice

```text
1, 2, 3, 4, 5, 6
```

Each is an outcome.

---

## 4. Sample Space

The **sample space** is the collection of all possible outcomes.

Usually represented by:

$$
S
$$

### Example — Coin

$$
S = \{H,T\}
$$

### Example — Dice

$$
S = \{1,2,3,4,5,6\}
$$

---

## 5. Event

An **event** is a specific outcome or group of outcomes that we are interested in.

Example:

Roll a dice.

Sample space:

$$
S=\{1,2,3,4,5,6\}
$$

Event A = getting an even number.

$$
A=\{2,4,6\}
$$

So there are 3 favorable outcomes.

---

# 6. Basic Probability Formula

For equally likely outcomes:

$$
P(A)=
\frac{\text{Number of Favorable Outcomes}}
{\text{Total Number of Outcomes}}
$$

### Example

Probability of getting an even number on a dice:

Favorable outcomes:

```text
2, 4, 6
```

Total outcomes:

```text
1, 2, 3, 4, 5, 6
```

Therefore:

$$
P(A)=\frac{3}{6}
$$

$$
P(A)=0.5
$$

So the probability is:

**50%**

---

# 7. Probability of Getting a 6

For a fair dice:

$$
P(6)=\frac{1}{6}
$$

Approximately:

$$
P(6)=0.1667
$$

or:

```text
16.67%
```

---

# 8. Complement of an Event

The **complement** means the event does **not** happen.

If event A is:

> Student passes

Then complement of A is:

> Student does not pass

It is represented as:

$$
A^c
$$

Formula:

$$
P(A^c)=1-P(A)
$$

### Example

Suppose:

$$
P(\text{Pass}) = 0.8
$$

Then:

$$
P(\text{Not Pass}) = 1 - 0.8 = 0.2
$$

So:

```text
Pass     = 80%
Not Pass = 20%
```

---

# 9. Probability Example — Students

Suppose a class has:

```text
100 students
```

Out of them:

```text
70 students passed
30 students failed
```

If one student is selected randomly:

$$
P(\text{Pass}) = \frac{70}{100} = 0.7
$$

Therefore:

```text
Probability of Pass = 70%
```

And:

$$
P(\text{Fail}) = \frac{30}{100} = 0.3
$$

---

# 10. Probability in Machine Learning

Probability is extremely important in ML.

Many ML models work with probabilities.

### Classification

Suppose a model predicts whether a customer will leave.

```text
Customer
   ↓
ML Model
   ↓
Churn Probability = 0.82
```

The model says:

$$
P(\text{Churn}) = 0.82
$$

or:

```text
82% probability of churn
```

If the classification threshold is `0.5`:

```text
0.82 > 0.5
```

Prediction:

```text
Churn = Yes
```

---

## 11. Probability vs Prediction

A classification model may produce:

| Customer | Churn Probability | Prediction |
|---|---:|---|
| A | 0.12 | No |
| B | 0.35 | No |
| C | 0.67 | Yes |
| D | 0.91 | Yes |

Probability gives us **confidence information**, while the final class depends on the chosen threshold.

---

# 12. Python Example

```python
passed = 70
total = 100

probability = passed / total

print(probability)
```

Output:

```text
0.7
```

Percentage:

```python
print(probability * 100)
```

Output:

```text
70.0
```

---

## 13. Important Probability Values

| Probability | Meaning |
|---:|---|
| `0` | Impossible |
| `0.1` | 10% chance |
| `0.25` | 25% chance |
| `0.5` | 50% chance |
| `0.75` | 75% chance |
| `0.9` | 90% chance |
| `1` | Certain |

---

## Key Points

- Probability measures **likelihood of an event**.
- Probability lies between **0 and 1**.
- `0` = impossible.
- `1` = certain.
- **Experiment** produces outcomes.
- **Outcome** = possible result.
- **Sample Space** = all possible outcomes.
- **Event** = outcome(s) we are interested in.
- Basic formula:

$$
P(A)=\frac{\text{Favorable Outcomes}}{\text{Total Outcomes}}
$$

- Complement:

$$
P(A^c)=1-P(A)
$$

- Probability is fundamental to **classification, Naive Bayes, logistic regression, uncertainty estimation, and many other ML concepts**.