
Note, this article is heavily inspired by [this original link]("https://hmkcode.github.io/ai/backpropagation-step-by-step/"), but all the weights and calculations have been changed to reflect a different perspective.

### Backpropagation Step by Step
<br>
Backpropagation is a commonly used technique to train neural networks.
Backpropagation stands for *"Backward propagation of gradients"*, it is a technique that uses, gradient descent to alter the weights and gently lead them towards a global minimum.

<br>
In this post, we will build a neural network with three layers:

- Input layer with two inputs neurons
- One hidden layer with two neurons
- Output layer with a single neuron
<br>

### Weights are everything
<br>
Neural network training is about finding weights that minimize prediction error. We usually start our training with a set of randomly generated weights.Then, backpropagation is used to update the weights in an attempt to correctly map arbitrary inputs to outputs.

Our initial weights will be as following: 
- w1 = 0.22
- w2 = 0.15
- w3 = 0.03
- w4 = 0.11
- w5 = 0.1
- w6 = 0.19

### Dataset
<br>
Our dataset is simple with 2 inputs and 1 output.
- Inputs = \[
\begin{bmatrix}
    i1  &  i2      \\
\end{bmatrix}
= 
\begin{bmatrix}
    2  &  3      \\
\end{bmatrix} 
\]
<br>
- Outputs=\\[
\begin{bmatrix}
    out  \\
\end{bmatrix}
= 
\begin{bmatrix}
    1      \\
\end{bmatrix} 
\\]

### Forward Pass
<br>
We will use given weights and inputs to predict the output. Inputs are multiplied by weights; the results are then passed forward to next layer.
<br>
<br>

\\[
\begin{bmatrix}
    2  &  3      \\
\end{bmatrix}
.
\begin{bmatrix}
    0.22  &  0.03      \\
    0.15  &  0.11      
\end{bmatrix} 
=
\begin{bmatrix}
    0.89  &  0.39      \\
\end{bmatrix}
\begin{bmatrix}
    0.1  \\
    0.19  \\
\end{bmatrix}
=
\begin{bmatrix}
    0.163  \\
\end{bmatrix}
\\]

<br>
- Calculation:     


$$ 2 * 0.22 + 3*0.15   =  0.89 \\
  2 * 0.03 + 3 * 0.11   = 0.39$$
  <br>
  <br>

 $$ 0.89 * 0.1 + 0.39 * 0.19 = 0.163$$
    
 

### Calculating Error
 <br>
 Now, it’s time to find out how our network performed by calculating the difference between the actual output and predicted one. It’s clear that our network output, or prediction, is not even close to actual output. We can calculate the difference or the error as following.
<br>

<br>
$$ \frac{1}{2} (prediction - actual)^2 $$
<br>

 $$ \frac{1}{2} (0.163 - 1)^2 = 0.3502$$
 
 
 

### Reducing Error
 <br>
 Our main goal of the training is to reduce the error or the difference between prediction and actual output. Since actual output is constant, “not changing”, the only way to reduce the error is to change prediction value. The question now is, how to change prediction value?

By decomposing prediction into its basic elements we can find that weights are the variable elements affecting prediction value. In other words, **in order to change prediction value, we need to change weights values.**
<br><br>

$$ prediction = out $$

$$ prediction = (h1).w5 + (h2).w6) $$

$$ prediction = (w1.i1 + w3.i2). w5) + (w2.i1 + w4.i2).w6) $$

<br>
 The question now is how to change\update the weights value so that the error is reduced?
The answer is **Backpropagation!**

### Backpropagation
<br>
We use backpropagation to allow the error to flow backwards throught the network. The correction is calculated via gradient descent.

$$ ^*W_x = W_x  - a(\frac{\delta(Error)}{\delta(W_x)})$$

where, 
- *Wx -> New Weight
- Wx -> Current Weight
- a -> Learning Rate

<br>
For example, to update *w6*, we take the current *w6* and subtract the partial derivative of error function with respect to *w6*. Optionally, we multiply the derivative of the error function by a selected number to make sure that the new updated weight is minimizing the error function; this number is called learning rate, and this number is chosen with some thought.

$$ ^*W_6 = W_6  - a(\frac{\delta(Error)}{\delta(W_6)})$$

The derivation of the error function is evaluated by applying the chain rule as following:
<br><br>

$$ \frac{\delta(Error)}{\delta(W_6)} = \frac{\delta(Error)}{\delta(prediction)} . \frac{\delta(prediction)}{\delta(W_6)}$$
<br>
<br>
$$ \frac{\delta(Error)}{\delta(W_6)} = \frac{1}{2}.(prediction - actual) . \frac{h_1W_5 + h_2W_6}{\delta(W_6)}$$
<br><br>

$$ \frac{\delta(Error)}{\delta(W_6)} = \frac{1}{2}*2.(prediction - actual).\frac{\delta(prediction - actual)}{\delta(prediction)} * h_2$$
<br>
<br>
$$ \frac{\delta(Error)}{\delta(W_6)} = (prediction - actual)* h_2,   \ \ \ \ \> \> \> \> \> \> \> \> \> \>\>\>\>\>\boxed{\Delta = (prediction - actual)}$$
<br>
<br>
$$ \boxed{\frac{\delta(Error)}{\delta(W_6)} = \Delta h_2} $$
<br>
<br>
So to update *w6* we can apply the following formula:
<br>
<br>
$$ ^*W_6 = W_6 - a\Delta h_2$$
<br>
<br>
Similarly, we can do the same for *w5*:
<br>
<br>
$$ ^*W_5 = W_5 - a\Delta h_1$$
<br><br>
However, when moving backward to update w1, w2, w3 and w4 existing between input and hidden layer, the partial derivative for the error function with respect to w1, for example, will be as following.
<br><br>
$$ \frac{\delta(Error)}{\delta(W_1)} = \frac{\delta(Error)}{\delta(prediction)}*\frac{\delta(prediction)}{\delta(h_1)}*\frac{\delta(h_1)}{\delta(W_1)}$$
<br><br>
$$ \frac{\delta(Error)}{\delta(W_1)} = \frac{\frac{1}{2}*(prediction - actual)^2}{\delta(prediction)}*\frac{\delta(h_1W_5 + h_2W_6)}{\delta(h_1)}*\frac{\delta(i_1W_1 + i_2W_2)}{\delta(W_1)}$$
<br><br>
$$ \frac{\delta(Error)}{\delta(W_1)} = 2*\frac{1}{2}(prediction - actual) \frac{\delta(prediction - actual)}{\delta(prediction)} * (W_5)* (i_1)$$
<br><br>
$$ \frac{\delta(Error)}{\delta(W_1)} = (prediction - actual) * (W_5i_1)  \ \ \ \ \> \> \> \> \> \> \> \> \> \>\>\>\>\>\boxed{\Delta = (prediction - actual)}$$
<br><br>
$$ \boxed{\frac{\delta(Error)}{\delta(W_1)} = \Delta W_5i_1}$$
<br><br>
In summary, the update formulas for all weights will be as following:
<br><br>
$$ ^*W_6 = W_6 - a(h_2. \Delta)$$
$$ ^*W_5 = W_5 - a(h_1. \Delta)$$
$$ ^*W_4 = W_4 - a(i_2. \Delta W_6)$$
$$ ^*W_3 = W_3 - a(i_1. \Delta W_6)$$
$$ ^*W_2 = W_2 - a(i_2. \Delta W_5)$$
$$ ^*W_1 = W_1 - a(i_1. \Delta W_5)$$
<br><br>

We can re-write the matrices as following:
<br><br>

\\[
\begin{bmatrix}
    W_5  \\
    W_6      \\
\end{bmatrix}
=
\begin{bmatrix}
    W_5  \\
    W_6      \\
\end{bmatrix}
-
a\Delta
\begin{bmatrix}
    h_1  \\
    h_2      \\
\end{bmatrix}
=
\begin{bmatrix}
    W_5  \\
    W_6      \\
\end{bmatrix}
-
\begin{bmatrix}
    a\Delta h_1  \\
    a\Delta h_2      \\
\end{bmatrix}
\\]
<br><br>
\\[
\begin{bmatrix}
    W_1 & W_3  \\
    W_2 & W_4      \\
\end{bmatrix}
= 
\begin{bmatrix}
    W_1 & W_3  \\
    W_2 & W_4      \\
\end{bmatrix}
-
a\Delta
\begin{bmatrix}
    i_1   \\
    i_2 \\
\end{bmatrix}
.
\begin{bmatrix}
    W_5  \\
    W_6     \\
\end{bmatrix}
=
\begin{bmatrix}
    W_1 & W_3  \\
    W_2 & W_4      \\
\end{bmatrix}
-
\begin{bmatrix}
    a\Delta i_1W_5 & a\Delta i_1W_6  \\
        a\Delta i_2W_5 & a\Delta i_2W_6  \\
\end{bmatrix}
\\]

### Backward Pass
<br>

Using derived formulas we can find the new weights.

> Learning rate: is a hyperparameter which means that we need to manually guess its value.
<br>

$\boxed{\Delta = 0.163 - 1 = -0.837}$
<br><br>
$taking, \ \boxed{a= 0.05}$
<br><br>

\\[
\begin{bmatrix}
    W_5  \\
    W_6      \\
\end{bmatrix}
=
\begin{bmatrix}
    0.1  \\
    0.19      \\
\end{bmatrix}
-
0.05(-0.837)
\begin{bmatrix}
    0.89  \\
       0.39\\
\end{bmatrix}
=
\begin{bmatrix}
    0.1  \\
    0.19      \\
\end{bmatrix}
-
\begin{bmatrix}
    -0.037  \\
    -0.0163      \\
\end{bmatrix}
=
\begin{bmatrix}
    0.14  \\
    0.21      \\
\end{bmatrix}
\\]
<br><br>

\\[
\begin{bmatrix}
    W_1 & W_3  \\
    W_2 & W_4      \\
\end{bmatrix}
= 
\begin{bmatrix}
    0.22 & 0.03  \\
    0.15  & 0.11     \\
\end{bmatrix}
-
0.05(-0.837)
\begin{bmatrix}
    2  \\
    3      \\
\end{bmatrix}
\begin{bmatrix}
    0.1 & 0.19  \\
\end{bmatrix}
=
\begin{bmatrix}
    0.22 & 0.03  \\
    0.15  & 0.11     \\
\end{bmatrix}
-
\begin{bmatrix}
    -0.0837 & -0.0159  \\
    -0.125  & -0.02385     \\
\end{bmatrix}
=
\begin{bmatrix}
    0.3 & 0.046  \\
    0.275  & 0.134     \\
\end{bmatrix}
\\]

<br>
Now, using the new weights we will repeat the forward pass:
<br><br>

\\[
\begin{bmatrix}
    2  & 3\\
\end{bmatrix}
=
\begin{bmatrix}
    0.3 & 0.046\\
    0.275 & 0.134
\end{bmatrix}
=
\begin{bmatrix}
    0.6825  & 0.49 \\
\end{bmatrix}
\begin{bmatrix}
    0.14\\
    0.21 \\
\end{bmatrix}
=
\begin{bmatrix}
    0.198 \\
\end{bmatrix}
\\]

<br>
<br>
$$ \ \ \boxed{original = 0.163} \> \> \> \boxed{after \ one\ backpropagation \ pass = 0.198}$$

<br>
We can notice that the prediction 0.198 is a little bit closer to actual output than the previously predicted one 0.163. We can repeat the same process of backward and forward pass until the error is close or equal to zero.


```python

```
