
# 30\. Matrices

## 1\. What is a matrix?

A **matrix** is a rectangular arrangement of numbers organized into rows and columns.

Example\:

$$
A=
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
$$

This matrix contains 2 rows and 3 columns.

Matrices are important in Machine Learning because datasets, model parameters, and many mathematical operations can be represented using matrices.

## 2\. Rows, columns, and dimensions

Consider the matrix\:

$$
A=
\begin{bmatrix}
10 & 20 & 30\\
40 & 50 & 60
\end{bmatrix}
$$

- **Rows\:** Horizontal arrangements of values.
- **Columns\:** Vertical arrangements of values.
- **Dimensions\:** Number of rows × number of columns.

The dimensions of $A$ are $2\times3$\.

### Understand matrix dimensions

$\begin{bmatrix}1&2\\3&4\end{bmatrix}$

2 × 2

$\begin{bmatrix}1&2&3\end{bmatrix}$

1 × 3

$\begin{bmatrix}1\\2\\3\end{bmatrix}$

3 × 1

A matrix with one row or one column is also a valid matrix.

## 3\. Types of matrices

| Type | Definition | Example |
|--------|--------|--------|
| Row matrix | Contains one row | $[1\ 2\ 3]$ |
| Column matrix | Contains one column | $\begin{bmatrix}1\\2\\3\end{bmatrix}$ |
| Square matrix | Rows = columns | $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ |
| Zero matrix | All elements are zero | $\begin{bmatrix}0&0\\0&0\end{bmatrix}$ |
| Identity matrix | Ones on the main diagonal, zeros elsewhere | $\begin{bmatrix}1&0\\0&1\end{bmatrix}$ |
| Diagonal matrix | Nonzero elements only on the main diagonal, possibly including zeros | $\begin{bmatrix}2&0\\0&5\end{bmatrix}$ |

The identity matrix is denoted by $I$\. Multiplying a compatible matrix by an identity matrix leaves it unchanged.

## 4\. Matrix operations

Consider two matrices\:

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},
\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}
$$

### A. Matrix addition

Add corresponding elements.

$$
A+B=
\begin{bmatrix}
1+5&2+6\\
3+7&4+8
\end{bmatrix}
=
\begin{bmatrix}6&8\\10&12\end{bmatrix}
$$

Both matrices must have the same dimensions.

### B. Matrix subtraction

Subtract corresponding elements.

$$
A-B=
\begin{bmatrix}-4&-4\\-4&-4\end{bmatrix}
$$

Both matrices must have the same dimensions.

### C. Scalar multiplication

Multiply every element by a scalar.

$$
3A=
\begin{bmatrix}3&6\\9&12\end{bmatrix}
$$

### D. Matrix multiplication

Matrix multiplication is different from multiplying corresponding elements.

If $A$ has dimensions $m\times n$, and $B$ has dimensions $n\times p$, then\:

$$
(m\times n)(n\times p)=(m\times p)
$$

The **inner dimensions must match**\.

For example\:

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},
\quad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}
$$

$$
AB=
\begin{bmatrix}
1(5)+2(7)&1(6)+2(8)\\
3(5)+4(7)&3(6)+4(8)
\end{bmatrix}
$$

$$
AB=\begin{bmatrix}19&22\\43&50\end{bmatrix}
$$

Notice that matrix multiplication is generally **not commutative**\: $AB$ need not equal $BA$\.

## 5\. Transpose of a matrix

The transpose switches rows and columns.

If\:

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
$$

Then\:

$$
A^T=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}
$$

The original dimensions are $2\times3$; the transposed dimensions are $3\times2$\.

## 6\. Implement matrices using NumPy

```python
import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print("Addition:\n", A + B)
print("Subtraction:\n", A - B)
print("Scalar multiplication:\n", 3 * A)
print("Matrix multiplication:\n", A @ B)
print("Transpose:\n", A.T)
print("Dimensions:", A.shape)
```

Output\:

```text
Addition:
 [[ 6  8]
  [10 12]]

Subtraction:
 [[-4 -4]
  [-4 -4]]

Scalar multiplication:
 [[ 3  6]
  [ 9 12]]

Matrix multiplication:
 [[19 22]
  [43 50]]

Transpose:
 [[1 3]
  [2 4]]

Dimensions: (2, 2)
```

**Important\:** In NumPy, `A @ B` performs matrix multiplication, while `A * B` performs element-wise multiplication.

## 7\. Why are matrices important in Machine Learning?

Suppose we have data for three students\:

| Student | Study hours | Attendance | Previous score |
|--------|--------|--------|--------|
| A | 2 | 60 | 55 |
| B | 4 | 75 | 70 |
| C | 6 | 90 | 85 |

The feature matrix is\:

$$
X=\begin{bmatrix}
2&60&55\\
4&75&70\\
6&90&85
\end{bmatrix}
$$

Its dimensions are $3\times3$\: three students and three features.

Matrices are used for\:

- Storing datasets as numerical arrays.
- Computing predictions for multiple samples efficiently.
- Representing weights in neural networks.
- Performing linear transformations.
- Calculating gradients during model training.

## Practice questions

1. Find the dimensions of a matrix with 4 rows and 5 columns.
2. Calculate the addition of $\begin{bmatrix}1&2\\3&4\end{bmatrix}$ and $\begin{bmatrix}5&6\\7&8\end{bmatrix}$\.
3. Find the transpose of $\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$\.
4. Can a $2\times3$ matrix be multiplied by a $3\times4$ matrix? What will be the output dimensions?
5. Explain the difference between `A * B` and `A @ B` in NumPy.
