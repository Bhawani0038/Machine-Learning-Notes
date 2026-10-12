# 36\. Gradient Descent

## 1\. What is Gradient Descent?

**Gradient Descent is an optimization algorithm used to minimize a loss function by repeatedly updating model parameters in the direction of the negative gradient.**

In Machine Learning, a model makes predictions, calculates its loss, and adjusts its parameters to reduce that loss.

For example, in Linear Regression, Gradient Descent helps find suitable values for the weight $w$ and bias $b$\.

The objective is\:

$$
\boxed{\text{Minimize } L(w,b)}
$$

Where $L$ represents the loss function.

## 2\. How Gradient Descent works

**1\. Initialize parameters**

Choose initial weights and bias.

**2\. Calculate predictions and loss**

Measure how far predictions are from actual values.

**3\. Calculate gradients**

Determine how the loss changes with each parameter.

**4\. Update parameters**

Move in the direction of decreasing loss.

**5\. Repeat**

Continue until a suitable stopping condition is reached.

## 3\. Gradient Descent formula

For a model parameter $w$, the update rule is\:

$$
\boxed{w_{\text{new}}=w_{\text{old}}-\alpha\frac{dL}{dw}}
$$

Where\:

- $w$ \= model parameter or weight
- $L$ \= loss function
- $\frac{dL}{dw}$ \= derivative of loss with respect to the weight
- $\alpha$ \= learning rate

For multiple parameters, the update rule becomes\:

$$
\boxed{\boldsymbol{\theta}_{\text{new}}
=\boldsymbol{\theta}_{\text{old}}-\alpha\nabla L(\boldsymbol{\theta})}
$$

Here, $\boldsymbol{\theta}$ represents the collection of model parameters.

**Why subtract the gradient?** The gradient points toward the direction of steepest local increase. Subtracting it moves the parameters in the direction of steepest local decrease.

## 4\. Numerical example: Minimize a function

Suppose\:

$$
L(w)=(w-4)^2
$$

The minimum loss occurs at $w=4$, where $L(w)=0$\.

### Step 1: Calculate the derivative

$$
\frac{dL}{dw}=2(w-4)
$$

### Step 2: Choose initial values

Let\:

$$
w=0,\qquad \alpha=0.1
$$

### Step 3: Calculate the gradient

$$
\frac{dL}{dw}=2(0-4)=-8
$$

### Step 4: Update the weight

$$
w_{\text{new}}=0-0.1(-8)
$$

$$
\boxed{w_{\text{new}}=0.8}
$$

The weight moves from 0 toward 4, reducing the loss from 16 to 10.24.

### Step 5: Repeat

The next update is\:

$$
w_{\text{new}}=0.8-0.1[2(0.8-4)]
$$

$$
w_{\text{new}}=1.44
$$

The weight continues moving toward 4. With a suitable learning rate, repeated updates approach the minimum.

## 5\. Interactive Gradient Descent demonstration

Change the learning rate and number of iterations to see how the parameter approaches the minimum of $L(w)=(w-4)^2$\.

## Loss function: $L(w)=(w-4)^2$

Learning rate ($\alpha$\)

> **Slider**
>
> Current value unavailable.
>
> **Range:** 0.01 to 1
>
> **Step:** 0.01
>
> **Scale labels:** 0.01 - 0\.01 · 0.5 - 0\.50 · 1 - 1\.00

Iterations

> **Slider**
>
> Current value unavailable.
>
> **Range:** 1 to 20
>
> **Step:** 1
>
> **Scale labels:** 1 - 1 · 10 - 10 · 20 - 20

---

Final weight

Final loss

### Optimization progress

Target weight: 4

0 4 (minimum)

### First few iterations

| Step | Weight | Loss |
|--------|--------|--------|

This is a one-parameter example. The green bar shows the final weight relative to the target weight of 4. A learning rate of 1 or higher does not converge to the minimum in this example.

[Reset demonstration]

## 6\. Learning rate: Why it matters

The learning rate controls the size of each parameter update.

| Learning rate | Typical behavior |
|--------|--------|
| Too small | Convergence may be very slow. |
| Suitable | Loss often decreases steadily toward a minimum. |
| Too large | Updates may overshoot, oscillate, or diverge. |

For the example $L(w)=(w-4)^2$, the update is\:

$$
w_{\text{new}}=w-2\alpha(w-4)
$$

When $0<\alpha<1$, the updates converge toward $w=4$\. When $\alpha=1$, the updates oscillate between opposite sides of the minimum without reducing the distance. When $\alpha>1$, the distance grows.

These specific thresholds apply to this quadratic function, not every loss function.

## 7\. Implement Gradient Descent using Python

```python
def loss(w):
    return (w - 4) ** 2

def gradient(w):
    return 2 * (w - 4)

learning_rate = 0.1
w = 0.0
iterations = 10

for step in range(iterations + 1):
    print(
        f"Step: {step:2d}, "
        f"Weight: {w:.4f}, "
        f"Loss: {loss(w):.4f}"
    )

    w = w - learning_rate * gradient(w)
```

Output\:

```text
Step:  0, Weight: 0.0000, Loss: 16.0000
Step:  1, Weight: 0.8000, Loss: 10.2400
Step:  2, Weight: 1.4400, Loss: 6.5536
Step:  3, Weight: 1.9520, Loss: 4.1943
Step:  4, Weight: 2.3616, Loss: 2.6844
Step:  5, Weight: 2.6893, Loss: 1.7180
Step:  6, Weight: 2.9514, Loss: 1.0995
Step:  7, Weight: 3.1611, Loss: 0.7037
Step:  8, Weight: 3.3289, Loss: 0.4504
Step:  9, Weight: 3.4631, Loss: 0.2883
Step: 10, Weight: 3.5705, Loss: 0.1845
```

The code calculates the gradient and updates the weight at every step. After 10 updates, the weight approaches 4 and the loss decreases.

## 8\. Types of Gradient Descent

| Type | How parameters are updated |
|--------|--------|
| Batch Gradient Descent | Uses the entire training dataset for each update. |
| Stochastic Gradient Descent (SGD) | Uses one training example for each update. |
| Mini-batch Gradient Descent | Uses a small batch of training examples for each update. |

Mini-batch Gradient Descent is widely used in modern neural network training because it balances computational efficiency with frequent updates.

## 9\. Limitations and practical considerations

- **Local minima and saddle points\:** The optimization path depends on the loss function and initialization.
- **Feature scaling\:** Features with very different scales can make optimization slower or less stable.
- **Learning rate selection\:** A poorly chosen learning rate can prevent effective convergence.
- **Stopping criteria\:** Training can stop after a fixed number of iterations or when improvement becomes sufficiently small.
- **Non-convex loss functions\:** Reaching a stationary point does not necessarily mean finding the global minimum.

## Practice questions

1. What is Gradient Descent, and why is it used in Machine Learning?
2. Write the parameter update formula.
3. For $L(w)=(w-3)^2$, calculate the gradient at $w=1$\.
4. If $w=1$, the gradient is $-4$, and the learning rate is $0.1$, calculate the updated weight.
5. What happens when the learning rate is too large?
6. Explain the difference between Batch, Stochastic, and Mini-batch Gradient Descent.
7. Why is feature scaling useful during gradient-based optimization?
