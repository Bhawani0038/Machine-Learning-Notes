# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 9: Events — Independent & Dependent Events

### What We Will Cover
- What is an Event?
- Types of Events
- Independent Events
- Dependent Events
- Mutually Exclusive Events
- Independent vs Dependent
- Real-Life Examples
- Probability Formulas
- ML Applications
- Key Points

---

## 1. What is an Event?

An **event** is an outcome or group of outcomes that we are interested in.

Example: Roll a dice.

$$
S=\{1,2,3,4,5,6\}
$$

Event A = getting an even number:

$$
A=\{2,4,6\}
$$

---

# 2. Independent Events

Two events are **independent** when the occurrence of one event **does not affect the probability of the other event**.

### Example: Tossing a Coin Twice

First toss:

```text
Head / Tail
```

Second toss:

```text
Head / Tail
```

The result of the first toss does not change the second toss.

Therefore:

> The two coin tosses are independent events.

---

## Formula

For independent events:

$$
P(A \cap B)=P(A)\times P(B)
$$

Where:

- $A \cap B$ = A and B both happen.

### Example

Probability of getting Head on first toss:

$$
P(A)=\frac{1}{2}
$$

Probability of getting Head on second toss:

$$
P(B)=\frac{1}{2}
$$

Therefore:

$$
P(A\cap B)
=
\frac12\times\frac12
=
\frac14
$$

So:

$$
P(\text{Head},\text{Head}) = 25\%
$$

---

# 3. Dependent Events

Two events are **dependent** when the occurrence of one event **affects the probability of the other event**.

### Example: Drawing Cards

Suppose we have a deck of cards.

We draw one card and **do not replace it**.

The first draw changes the number of cards available for the second draw.

Therefore:

> The events are dependent.

---

## Example

Probability of drawing an Ace first:

$$
P(A)=\frac{4}{52}
$$

If the first card was an Ace and it is not replaced:

Remaining cards:

$$
51
$$

Remaining Aces:

$$
3
$$

Therefore:

$$
P(B \mid A)=\frac{3}{51}
$$

Notice that the probability of the second event changed.

---

# 4. Dependent Event Formula

For dependent events:

$$
P(A\cap B)=P(A)\times P(B \mid A)
$$

Where:

$$
P(B \mid A)
$$

means:

> Probability of B **given that A has already happened**.

This leads directly to the concept of **conditional probability**.

---

# 5. Independent vs Dependent

| Independent | Dependent |
|---|---|
| One event does not affect another | One event affects another |
| Probability remains unchanged | Probability changes |
| Coin tosses | Drawing cards without replacement |
| Rolling dice | Selecting students without replacement |
| $P(B \mid A)=P(B)$ | $P(B \mid A)\neq P(B)$ |

---

# 6. Mutually Exclusive Events

Two events are **mutually exclusive** if they **cannot happen at the same time**.

Example: Roll one dice.

Event A:

```text
Getting 2
```

Event B:

```text
Getting 5
```

You cannot get both 2 and 5 in the **same single dice roll**.

Therefore:

$$
P(A\cap B)=0
$$

---

## Probability of A OR B

For mutually exclusive events:

$$
P(A\cup B)=P(A)+P(B)
$$

Example:

Probability of getting 2 or 5:

$$
P(2)=\frac16
$$

$$
P(5)=\frac16
$$

Therefore:

$$
P(2 \text{ or } 5)
=
\frac16+\frac16
=
\frac26
=
\frac13
$$

---

# 7. Important Difference

Don't confuse:

### Independent

Events **can happen together**, but one does not affect the other.

Example:

```text
Coin Toss 1 = Head
Coin Toss 2 = Head
```

### Mutually Exclusive

Events **cannot happen together**.

Example:

```text
One dice roll = 2 AND 5
```

Impossible.

---

# 8. Real-Life Example — Student Selection

Suppose a class contains:

```text
60 students
```

- 30 male
- 30 female

You select one student.

Then select another student **without replacement**.

The first selection changes the number of students remaining.

Therefore:

```text
Dependent Events
```

But if you select a student, replace them, and then select another student:

```text
Independent Events
```

because the original probabilities remain unchanged.

---

# 9. Machine Learning Connection

Events and their relationships are important in ML probability concepts.

For example:

```text
Customer
   ↓
Purchased Product
   ↓
Purchased Premium Product
```

We may want to calculate:

$$
P(\text{Premium Product} \mid \text{Purchased})
$$

This is a **conditional probability**.

It is heavily used in:

- Naive Bayes
- Bayesian inference
- Risk prediction
- Medical diagnosis
- Spam detection
- Fraud detection

---

## Key Points

- An **event** is an outcome or group of outcomes.
- **Independent events** do not affect each other's probability.
- Independent formula:

$$
P(A\cap B)=P(A)P(B)
$$

- **Dependent events** affect each other's probability.
- Dependent formula:

$$
P(A\cap B)=P(A)P(B \mid A)
$$

- **Mutually exclusive events** cannot occur together.
- For mutually exclusive events:

$$
P(A\cap B)=0
$$

- Independent ≠ mutually exclusive.
- Conditional probability is the next important concept for ML and leads to **Bayes Theorem**.