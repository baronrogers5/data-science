### Name -Piyush Daga

### Batch - 4

[Assignment 2A](https://github.com/baronrogers5/eip_assignments/blob/master/python_101.ipynb)


## Assignment 2B - Backprop Calculation
---

[python file wth computations](https://github.com/baronrogers5/eip_assignments/blob/master/python_computations.ipynb)
####  Step 0: Read Input and Output
``` python
import numpy as np
# random initialization
X = np.array( [ [1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 0] ])
````

| X ||||
--- | --- | --- | ---
1 | 0 | 1 | 1 
0 | 1| 0| 0
1| 0|1|0

#### Step 1: Initialize weights and biases with random values
``` python
wh = np.round(np.random.random(size=(4,3)), decimals=2)
bh = np.round(np.random.random(size=(1,3)), decimals=2)
````


| X |||| wh |||bh ||| 
:---: | --- | --- | --- | :---: | --- | --- | --- | ---|---|
1 | 0 | 1 | 1 | 0.1 |0.81 |0.3 |0.5 | 0.24 |  0.47
0 | 1| 0| 0 |0.26 | 0.29 | 0.2
1 | 0 | 1 | 0 |0.4 | 0.69|  0.81
|||| |  0.2 |  0.37|  0.35

#### Step 2: Calcluate hidden layer input
``` python
hidden_layer_input = np.dot(X, wh) + bh
````

| X |||| wh |||bh ||| hidden_layer_input |||
:---: | --- | --- | --- | :---: | --- | --- | --- | ---|---|:---:|---|---|
1 | 0 | 1 | 1 | 0.1 |0.81 |0.3 |0.5 | 0.24 |  0.47 |1.2 |  2.11 | 1.93
0 | 1| 0| 0 |0.26 | 0.29 | 0.2 | ||| 0.76|  0.53|  0.67
1 | 0 | 1 | 0 |0.4 | 0.69|  0.81 | ||| 1.|1.74 | 1.58
|||| |  0.2 |  0.37|  0.35| 

#### Step 3: hidden layer activations
``` python
sigmoid = lambda x: np.round(1/(1 + np.exp(-x)), decimals=2)
hiddenlayer_activations = sigmoid(hidden_layer_input)
````
| X |||| wh |||bh ||| hidden_layer_input ||| hidden_layer_activations |||
:---: | --- | --- | --- | :---: | --- | --- | :---: | ---|---|:---:|---|---| :---: |---|---|
1 | 0 | 1 | 1 | 0.1 |0.81 |0.3 |0.5 | 0.24 |  0.47 |1.2 |  2.11 | 1.93 |0.77| 0.89|  0.87
0 | 1| 0| 0 |0.26 | 0.29 | 0.2 | ||| 0.76|  0.53|  0.67 |0.68|  0.63|  0.66
1 | 0 | 1 | 0 |0.4 | 0.69|  0.81 | ||| 1.|1.74 | 1.58 |  0.73 | 0.85| 0.83
|||| |  0.2 |  0.37|  0.35| 

#### Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer
``` python
wout = np.round(np.random.random(size=(3,1)), decimals=2)
bout = np.round(np.random.random(size=(1,1)), decimals=2)
output_layer_input = np.dot(hiddenlayer_activations, wout) + bout
output = sigmoid(output_layer_input)
```
| wout | bout | output | y |
| :---: | :---: | :---: | :---: |
| 0.55 | 0.58 | 0.86 | 1
| 0.78 | | 0.83 | 1
| 0.18 | | 0.86 | 0

#### Step 5: Calculate Error
``` python
E = y - output
````
| E |
| :---: |
| 0.14 |
| 0.17 |
| -0.86|

#### Step 6: Compute slope at output and hidden layer
``` python 
derivaties_sigmoid = lambda x : x*(1 - x)
slope_output_layer = np.round(derivaties_sigmoid(output), 2)
slope_hidden_layer = np.round(derivaties_sigmoid(hiddenlayer_activations), 2)
````
| slope_output_layer | slope_hidden_layer|||
| :---: | :---: | --- | ---|
| 0.12 | 0.18| 0.1 |0.11
| 0.14 |0.22|0.23|0.22
| 0.12 |0.2 |0.13|0.14

#### Step 7: Compute delta at output layer
``` python
lr = 1
delta_output = np.round(E * slope_output_layer * lr, 2)
```
| delta output |
| :---: |
|0.02|
|0.02|,
 | -0.1|

#### Step 8: Error at hidden layer
``` python
error_at_hidden_layer = np.round(np.dot(delta_output, wout.T),2)
error_at_hidden_layer
````

| Error at hidden layer |||
|:---: |---|---|
|0.01 |0.02 | 0.  |
| 0.01|  0.02|  0. |
|-0.06| -0.08| -0.02|

#### Step 9: Compute delta at hidden layer
``` python
d_hidden = np.round(error_at_hidden_layer * slope_hidden_layer,2)
````

| d_hidden|||
|:---:|---| ---|
|0|0|0
|0|0|0
|-0.01|-0.01|0

#### Step 10: Update the weights of both output and hidden layer
``` python 
wout = np.round(wout + np.dot(hiddenlayer_activations.T, delta_output), 2)
wh = np.round(wh + np.dot(X.T, d_hidden), 2)
````

| wh ||| wout |
| :---: | --- | ---| ---|
|0.09 | 0.8 | 0.3 | 0.51|
| 0.26|  0.29|  0.2 | 0.73|
| 0.39|  0.68|  0.81 |0.13
|0.2  |0.37 |0.35|

#### Step 11: Update the biases
``` python 
bh = bh + np.sum(d_hidden, axis = 0)
bout = bout + np.sum(delta_output, axis=0)
````
| bh||| bout |
|:---:|---|---|:---:|
|0.49 | 0.23 |0.47|0.52









