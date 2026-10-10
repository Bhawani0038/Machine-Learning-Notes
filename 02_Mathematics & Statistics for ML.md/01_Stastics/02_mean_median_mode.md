# Module 2 — Mathematics & Statistics for Machine Learning
## Topic 2: Mean, Median, and Mode

### What We Will Cover

- Mean
- Median
- Mode
- How to calculate them
- Mean vs Median vs Mode
- Effect of outliers
- Real-life Data Science examples
- When to use each measure
- Key Points

---

# 1. Mean

**Mean is the average value of a dataset.**

### Formula

$$
\text{Mean} = \frac{\text{Sum of all values}}{\text{Number of values}}
$$

### Example

Suppose student marks are:

```text
10, 20, 30, 40, 50
```

Sum:

```text
10 + 20 + 30 + 40 + 50 = 150
```

Number of values:

```text
5
```

Therefore:

$$
Mean = \frac{150}{5} = 30
$$

So:

**Mean = 30**

---

# 2. Mean in Python

```python
import numpy as np

marks = np.array([10, 20, 30, 40, 50])

print(np.mean(marks))
```

Output:

```text
30.0
```

---

# 3. Median

**Median is the middle value when the data is arranged in sorted order.**

Example:

```text
10, 20, 30, 40, 50
```

The middle value is:

```text
30
```

Therefore:

**Median = 30**

---

# 4. Median with Even Number of Values

Suppose:

```text
10, 20, 30, 40
```

There are two middle values:

```text
20 and 30
```

Median:

$$
Median = \frac{20 + 30}{2}
$$

$$
Median = 25
$$

So:

**Median = 25**

---

# 5. Median in Python

```python
import numpy as np

marks = np.array([10, 20, 30, 40, 50])

print(np.median(marks))
```

Output:

```text
30.0
```

---

# 6. Mode

**Mode is the value that occurs most frequently in a dataset.**

Example:

```text
10, 20, 20, 30, 40
```

Frequency:

```text
10 → 1
20 → 2
30 → 1
40 → 1
```

Therefore:

**Mode = 20**

---

# 7. Mode in Python

Using Pandas:

```python
import pandas as pd

marks = pd.Series([10, 20, 20, 30, 40])

print(marks.mode())
```

Output:

```text
0    20
dtype: int64
```

---

# 8. Mean vs Median vs Mode

| Measure | Meaning | Example |
|---|---|---:|
| Mean | Average | 30 |
| Median | Middle value | 30 |
| Mode | Most frequent value | 20 |

---

# 9. The Important Effect of Outliers

Consider salaries:

```text
30,000
35,000
40,000
45,000
5,00,000
```

The last value is much larger than the others.

Let's calculate the mean:

$$
Mean = \frac{30000+35000+40000+45000+500000}{5}
$$

$$
Mean = 130000
$$

So:

**Mean = ₹1,30,000**

But most employees earn between ₹30,000 and ₹45,000.

Now sort the values:

```text
30,000
35,000
40,000
45,000
5,00,000
```

The middle value is:

**₹40,000**

Therefore:

```text
Mean   = ₹1,30,000
Median = ₹40,000
```

The extreme value has strongly affected the **mean**, but the median is much less affected.

---

# 10. Why Median Is Useful

Median is particularly useful when the data is **skewed** or contains extreme values.

### Example — House Prices

Suppose house prices are:

```text
30 lakh
35 lakh
40 lakh
45 lakh
50 lakh
5 crore
```

The 5-crore house can pull the mean upward.

For understanding the **typical house price**, median may give a more representative value.

---

# 11. When to Use Mean

Use mean when:

- Data is reasonably symmetric.
- There are no major outliers.
- You want the mathematical average.
- Numerical calculations require an average.

Examples:

- Average marks
- Average temperature
- Average monthly sales
- Average production

---

# 12. When to Use Median

Use median when:

- Data contains significant outliers.
- Data is highly skewed.
- You want the middle/typical value.

Examples:

- House prices
- Salaries
- Income
- Property values

---

# 13. When to Use Mode

Mode is useful when you want to know the **most common value/category**.

Examples:

### Most common shoe size

```text
7, 8, 8, 8, 9, 10
```

Mode:

```text
8
```

### Most common city

```text
Delhi
Mumbai
Delhi
Jaipur
Delhi
```

Mode:

```text
Delhi
```

Mode is especially useful for **categorical data**, where mean and median may not make sense.

---

# 14. Simple Data Science Example

Suppose customer order amounts are:

```text
500
700
800
900
1000
25000
```

We can calculate:

```text
Mean   → strongly affected by 25,000
Median → represents the middle order better
Mode   → useful only if repeated values exist
```

This is why Data Scientists don't automatically use mean everywhere.

They first understand the **distribution of the data**.

---

## Quick Comparison

```text
Mean
↓
Average

Median
↓
Middle value

Mode
↓
Most frequent value
```

---

## Key Points

- **Mean = Average**
- **Median = Middle value**
- **Mode = Most frequent value**
- Mean is sensitive to **outliers**.
- Median is more robust to extreme values.
- Mode is useful for finding the most common value/category.
- For skewed data such as **salary and house prices**, median can be more representative than mean.
- Always understand the data before choosing a measure of central tendency.