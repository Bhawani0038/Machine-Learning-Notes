# Module 2 — Mathematics & Statistics for Machine Learning

## Topic 11: Bayes Theorem

### What We Will Cover
- What is Bayes Theorem?
- Why Bayes Theorem is Needed
- Formula
- Prior Probability
- Likelihood
- Evidence
- Posterior Probability
- Simple Student Example
- Medical Example
- ML Connection
- Key Points

---

# 1. What is Bayes Theorem?

**Bayes Theorem helps us calculate the probability of an event when we have new information or evidence.**

It allows us to **update our belief based on new evidence**.

For example:

> A patient tests positive. What is the probability that the patient actually has the disease?

We want:

$$
P(\text{Disease} \mid \text{Positive})
$$

Bayes Theorem helps us calculate this.

---

# 2. Bayes Theorem Formula

$$
\boxed{
P(A \mid B)=
\frac{P(B \mid A)P(A)}
{P(B)}
}
$$

Where:

| Term | Meaning |
|---|---|
| $P(A \mid B)$ | Posterior |
| $P(B \mid A)$ | Likelihood |
| $P(A)$ | Prior |
| $P(B)$ | Evidence |

The most important idea:

$$
\boxed{\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}}
$$

---

# 3. Understanding the Terms

Suppose:

```text
A = Disease
B = Positive Test
```

### Prior

$$
P(Disease)
$$

Probability of disease **before seeing the test result**.

---

### Likelihood

$$
P(\text{Positive} \mid \text{Disease})
$$

Probability of getting a positive test **when the person actually has the disease**.

---

### Evidence

$$
P(Positive)
$$

Overall probability of getting a positive test.

---

### Posterior

$$
P(\text{Disease} \mid \text{Positive})
$$

Probability that the person actually has the disease **after knowing that the test is positive**.

---

# 4. Student Example

Suppose:

- 60% students study regularly.
- 80% of regular students pass.
- 30% of irregular students pass.

We want:

> If a randomly selected student passed, what is the probability that they studied regularly?

We need:

$$
P(\text{Regular} \mid \text{Pass})
$$

---

## Step 1 — Known Probabilities

$$
P(Regular)=0.60
$$

$$
P(\text{Pass} \mid \text{Regular})=0.80
$$

$$
P(Irregular)=0.40
$$

$$
P(\text{Pass} \mid \text{Irregular})=0.30
$$

---

## Step 2 — Calculate Probability of Passing

A student can pass through two groups:

```text
Regular → Pass
Irregular → Pass
```

Therefore:

$$
P(Pass)
=
P(\text{Pass} \mid \text{Regular})P(Regular)
+
P(\text{Pass} \mid \text{Irregular})P(Irregular)
$$

Substitute:

$$
P(Pass)
=
(0.80)(0.60)+(0.30)(0.40)
$$

$$
=0.48+0.12
$$

$$
=0.60
$$

So:

$$
P(Pass)=0.60
$$

---

## Step 3 — Apply Bayes Theorem

$$
P(\text{Regular} \mid \text{Pass})
=
\frac{P(\text{Pass} \mid \text{Regular})P(Regular)}
{P(Pass)}
$$

$$
=
\frac{0.80\times0.60}{0.60}
$$

$$
=
\frac{0.48}{0.60}
$$

$$
=0.80
$$

### Answer:

$$
\boxed{P(\text{Regular} \mid \text{Pass})=80\%}
$$

---

# 5. Medical Example

Suppose:

- Disease prevalence = **5%**
- Test correctly detects disease = **95%**
- False positive rate = **10%**

We want:

$$
P(\text{Disease} \mid \text{Positive})
$$

### Given:

$$
P(D)=0.05
$$

$$
P(\text{Positive} \mid D)=0.95
$$

Healthy population:

$$
P(NoD)=0.95
$$

False positive:

$$
P(\text{Positive} \mid \text{NoD})=0.10
$$

---

### Evidence

$$
P(\text{Positive})
=
(0.95)(0.05)+(0.10)(0.95)
$$

$$
=0.0475+0.095
$$

$$
=0.1425
$$

---

### Bayes Theorem

$$
P(D \mid \text{Positive})
=
\frac{0.95\times0.05}{0.1425}
$$

$$
\approx0.3333
$$

Therefore:

$$
\boxed{P(\text{Disease} \mid \text{Positive})\approx33.33\%}
$$

### Important Intuition

Even though the test detects **95% of diseased people**, a positive result does **not** mean there is a 95% chance the person has the disease.

Why?

Because the disease is relatively rare and there are also false positives.

---

# 6. Bayes Theorem Intuition

Think of it as:

```text
OLD BELIEF
    ↓
New Evidence
    ↓
UPDATE
    ↓
NEW BELIEF
```

Mathematically:

```text
Prior
  ↓
Evidence
  ↓
Posterior
```

---

# 7. Bayes Theorem in Machine Learning

Bayes Theorem is the foundation of **Bayesian reasoning** and is directly used in **Naive Bayes** classification.

For example, spam detection:

```text
Email
 ↓
Words/features
 ↓
Probability of Spam
 ↓
Probability of Not Spam
 ↓
Final Prediction
```

The model estimates:

$$
P(\text{Spam} \mid \text{Words})
$$

and:

$$
P(\text{Not Spam} \mid \text{Words})
$$

Then selects the more probable class.

---

## Key Points

- Bayes Theorem **updates probability using new evidence**.
- Formula:

$$
\boxed{
P(A \mid B)=
\frac{P(B \mid A)P(A)}
{P(B)}
}
$$

- **Prior** = probability before evidence.
- **Likelihood** = probability of evidence given the event.
- **Evidence** = overall probability of the evidence.
- **Posterior** = updated probability after evidence.
- $P(A \mid B)$ and $P(B \mid A)$ are generally **not the same**.
- Bayes Theorem is fundamental to **Naive Bayes** and many probabilistic ML methods.