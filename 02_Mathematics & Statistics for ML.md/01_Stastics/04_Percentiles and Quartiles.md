# Module 2 — Mathematics & Statistics for Machine Learning
## Topic 4: Percentiles and Quartiles

### What We Will Cover

- Percentile
- Quartile
- Q1, Q2, Q3
- Median and Q2
- Interquartile Range (IQR)
- Five-number summary
- Real-life examples
- Outlier detection intuition
- Python implementation
- Key Points

---

# 1. What is a Percentile?

A **percentile tells us the relative position of a value within a dataset**.

For example:

> If a student's score is at the **90th percentile**, the student performed better than approximately 90% of the students in the reference dataset.

It does **not** mean the student scored 90%.

### Example

Suppose a student is at the:

```text
90th percentile
```

This means the student's score is higher than approximately 90% of the observations.

---

# 2. Simple Example

Suppose the sorted marks are:

```text id="5x9b9d"
10, 20, 30, 40, 50,
60, 70, 80, 90, 100
```

Percentiles divide the distribution according to position.

Commonly used percentiles:

```text id="u4q1tq"
25th percentile
50th percentile
75th percentile
90th percentile
95th percentile
99th percentile
```

---

# 3. Quartiles

**Quartiles divide ordered data into four parts.**

There are three important quartiles:

### Q1 — First Quartile

25th percentile:

$$
Q1 = P_{25}
$$

Approximately 25% of observations are below Q1.

---

### Q2 — Second Quartile

50th percentile:

$$
Q2 = P_{50}
$$

Q2 is the **median**.

$$
Q2 = Median
$$

---

### Q3 — Third Quartile

75th percentile:

$$
Q3 = P_{75}
$$

Approximately 75% of observations are below Q3.

---

# 4. Quartile Diagram

```text id="s0jv4m"
0%                 25%        50%        75%                100%
│-------------------│-----------│-----------│------------------│
                    Q1          Q2          Q3
                                ↑
                              Median
```

So:

```text id="x3x4cw"
Q1 = 25th percentile
Q2 = 50th percentile = Median
Q3 = 75th percentile
```

---

# 5. Example

Consider:

```text id="w2f8jm"
10, 20, 30, 40, 50,
60, 70, 80, 90
```

For this dataset, the approximate quartiles are:

```text id="8kdb2x"
Q1 ≈ 30
Q2 = 50
Q3 ≈ 70
```

The exact percentile calculation can vary slightly depending on the percentile method used by a software library.

---

# 6. Interquartile Range — IQR

The **Interquartile Range (IQR)** measures the spread of the middle 50% of the data.

### Formula

$$
IQR = Q3 - Q1
$$

If:

```text id="h6d1o9"
Q1 = 30
Q3 = 70
```

Then:

$$
IQR = 70 - 30 = 40
$$

So:

**IQR = 40**

---

# 7. Why Is IQR Important?

IQR focuses on the **middle 50% of the data**.

It is less affected by extreme values than the range.

Example:

```text id="q8d2ll"
20, 30, 40, 50, 60, 70, 80
```

Now add an extreme value:

```text id="c9d9b2"
20, 30, 40, 50, 60, 70, 80, 1000
```

The range changes dramatically because of `1000`.

But IQR is much more resistant to such extreme observations.

---

# 8. IQR and Outlier Detection

A common statistical rule uses IQR to identify potential outliers.

### Lower Bound

$$
Lower = Q1 - 1.5 \times IQR
$$

### Upper Bound

$$
Upper = Q3 + 1.5 \times IQR
$$

Values below the lower bound or above the upper bound are commonly flagged as **potential outliers**.

---

# 9. Example of IQR Outlier Detection

Suppose:

```text id="r2r8ba"
Q1 = 20
Q3 = 40
```

First:

$$
IQR = 40 - 20 = 20
$$

Lower bound:

$$
20 - 1.5(20) = -10
$$

Upper bound:

$$
40 + 1.5(20) = 70
$$

Therefore:

```text id="5b5e4c"
Values < -10 → Potential outlier
Values > 70  → Potential outlier
```

If our dataset contains:

```text id="h7y7cr"
10, 20, 25, 30, 35, 40, 100
```

then `100` would be flagged as a potential outlier.

---

# 10. Five-Number Summary

A very useful statistical summary consists of five values:

```text id="j2m9ll"
Minimum
Q1
Median
Q3
Maximum
```

Example:

```text id="3n8kfg"
Minimum = 10
Q1      = 25
Median  = 40
Q3      = 65
Maximum = 100
```

This is called the **Five-Number Summary**.

It is the foundation of a **box plot**.

---

# 11. Python — Percentiles

Using NumPy:

```python id="q0gq4q"
import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

print("25th Percentile:", np.percentile(data, 25))
print("50th Percentile:", np.percentile(data, 50))
print("75th Percentile:", np.percentile(data, 75))
```

---

# 12. Python — Quartiles and IQR

```python id="k7c1eq"
import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)
Q3 = np.percentile(data, 75)

IQR = Q3 - Q1

print("Q1:", Q1)
print("Q2:", Q2)
print("Q3:", Q3)
print("IQR:", IQR)
```

Output:

```text id="d0s1m5"
Q1: 30.0
Q2: 50.0
Q3: 70.0
IQR: 40.0
```

---

# 13. Real-World Example — Salaries

Suppose a company analyzes employee salaries.

It finds:

```text id="b6d6bs"
Q1 = ₹30,000
Median = ₹45,000
Q3 = ₹70,000
```

Interpretation:

- Approximately 25% of employees earn below ₹30,000.
- Approximately 50% earn below ₹45,000.
- Approximately 75% earn below ₹70,000.
- The middle 50% lies roughly between ₹30,000 and ₹70,000.

This is much more informative than looking only at the average.

---

## Key Points

- **Percentile → position relative to the distribution.**
- **Q1 = 25th percentile.**
- **Q2 = 50th percentile = Median.**
- **Q3 = 75th percentile.**
- **IQR = Q3 − Q1.**
- IQR represents the spread of the **middle 50%**.
- IQR is commonly used for **potential outlier detection**.
- Five-number summary = **Min, Q1, Median, Q3, Max**.
- Percentiles and quartiles are widely used in **EDA, statistics, and ML preprocessing**.