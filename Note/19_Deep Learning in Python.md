# Deep Learning in Python

## 1. Basics of deep learning and neural networks
> * familiar with the fundamental concepts and terminology used in deep learning
> * build simple neural networks yourself and generate predictions with them

### Introduction to deep learning
> #### Comparing neural network models to classical regression models
> > * more nodes in the hidden layer, and therefore, greater ability to capture interactions

![113](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/113.png)

### Forward propagation
> #### Coding the forward propagation algorithm
> > * write code to do forward propagation (prediction) for your first neural network:

![114](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/114.png)

```python
# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)  # -39
```

### Activation functions
> #### The Rectified Linear Activation Function
> > * "activation function" is a function applied at each node. It converts the node's input into some output.
> > * The rectified linear activation function (called ReLU) has been shown to lead to very high-performance networks. 
> > * This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.
> > * Without this activation function, you would have predicted a negative number! 
> > * The real power of activation functions will come soon when you start tuning model weights.

```python
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output) # 52
```

> #### Applying the network to many observations/rows of data

```python
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)

```

### Deeper networks
> #### Multi-layer neural networks

![115](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/115.png)

```python
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = relu((hidden_1_outputs * weights['output']).sum())
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)  # 364
```


## 2. Optimizing a neural network with backward propagation
> * how to optimize the predictions generated by your neural networks. You'll do this using a method called backward propagation, which is one of the most important techniques in deep learning

### The need for optimization
> #### Coding how weight changes affect accuracy
> > * change weights in a real network and see how they affect model accuracy!

```python
# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [-1, 1]
            }

# Make prediction using new weights: model_output_1
model_output_1 = 3

# Calculate error: error_1
error_1 = 3 - 3

# Print error_0 and error_1
print(error_0) # 6
print(error_1) # 0
```

![116](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/116.png)

> #### Scaling up to multiple data points
> > * use the mean_squared_error() function from sklearn.metrics. It takes the true values and the predicted values as arguments.

```python
from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row,weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row,weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0) # 294
print("Mean squared error with weights_1: %f" %mse_1) # 395
```

### Gradient descent
> #### Calculating slopes
> > * the slope is 2 * x * (y-xb), or 2 * input_data * error
> > * x and b may have multiple numbers (x is a vector for each data point, and b is a vector). 

```python
# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)  # [14 28 42]
```

> #### Improving model weights
> > * use those slopes to improve your model. 
> > * If you add the slopes to your weights, you will move in the right direction. However, 
> > * it's possible to move too far in that direction. So you will want to take a small step in that direction first, using a lower learning rate, and verify that the model is improving.

```python
# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - learning_rate * slope

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error) # 7

# Print the updated error
print(error_updated) # 5.04
```

> #### Making multiple updates to weights
> > * make multiple updates so you can dramatically improve your model weights, and see how the predictions improve with each update.
> > * pre-loaded get_slope() function that takes input_data, target, and weights as arguments. There is also a get_mse() function that takes the same arguments. The input_data, target, and weights have been pre-loaded.

```python
n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
```
![117](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/117.png)

### Backpropagation
> #### The relationship between forward and backward propagation
> > * Each time you generate predictions using forward propagation, you update the weights using backward propagation.

> #### Thinking about backward propagation
> > * if your predictions were all exactly right, and your errors were all exactly 0, the slope of the loss function with respect to your predictions would also be 0. 
> > * the updates to all weights in the network would indeed also be 0

### Backpropagation in practice
> #### A round of backpropagation
> > * node values calculated as part of forward propagation are shown in white.
> > * The weights are shown in black. 
> > * Layers after the question mark show the slopes calculated as part of back-prop, rather than the forward-prop values. Those slope values are shown in purple.
> > * uses the ReLU activation function, so the slope of the activation function is 1 for any node receiving a positive value as input. Assume the node being examined had a positive value (so the activation function's slope is 1).

![118](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/118.png)

![119](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/119.png)

> > * Ans: 6

## 3. Building deep learning models with keras
> * use the keras library to build deep learning models for both regression as well as classification
> * learn about the Specify-Compile-Fit workflow that you can use to make predictions

### Creating a keras model
> #### Specifying a model
> > * take the skeleton of a neural network and add a hidden layer and an output layer. You'll then fit that model and see Keras do the optimization so your model continually gets better.

```python
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32,  activation='relu'))

# Add the output layer
model.add(Dense(1))
```

### Compiling and fitting a model
> #### Compiling the model
> > * To compile the model, you need to specify the optimizer and loss function to use. 
> > * the Adam optimizer is an excellent choice
> > * https://keras.io/optimizers/#adam

```python
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss) # Loss function: mean_squared_error

```

> #### Fitting the model
> > * the data to be used as predictive features is loaded in a NumPy matrix called predictors 
> > * the data to be predicted is stored in a NumPy matrix called target

```python
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)
```

### Classification models
> #### Last steps in classification models
> > *  you'll use the 'sgd' optimizer, which stands for Stochastic Gradient Descent.
> > * https://en.wikipedia.org/wiki/Stochastic_gradient_descent
> > * output layer, because it is a classification model, the activation should be 'softmax'.
> > * metrics=['accuracy'] to see the accuracy (what fraction of predictions were correct) at the end of each epoch.

```python
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)
```
![120](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/120.png)

### Using models
> #### Saving, reloading and using your Model

```python
from keras.models import load_modelmodel.save('model_file.h5')my_model = load_model('my_model.h5')predictions = my_model.predict(data_to_predict_with)probability_true = predictions[:,1]
```

```python
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)

```

## 4. Fine-tuning keras models
> * how to optimize your deep learning models in keras. You'll learn how to validate your models, understand the concept of model capacity, and experiment with wider and deeper networks.

### Understanding model optimization
> #### Changing optimization parameters
> > *  try optimizing a model at a very low learning rate, a very high learning rate, and a "just right" learning rate. 
> > *  remembering that a low value for the loss function is good.

```python
# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer= my_optimizer, loss= 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)
```

### Model validation
> #### Evaluating model accuracy on validation dataset
> > * Create a validation split of 30% (or 0.3)

```python
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors, target, validation_split=0.3)
```

> #### Early stopping: Optimizing the optimization
> > * use early stopping to stop optimization when it isn't helping any more
> > * Stop optimization when the validation loss hasn't improved for 2 epochs by specifying the patience parameter of EarlyStopping() to be 2.

```python
# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, validation_split=0.3, epochs=30 ,callbacks=[early_stopping_monitor])
```
> #### Experimenting with wider networks
> > * create a new model called model_2 which is similar to model_1, except it has 100 units in each hidden layer
> > * added the argument verbose=False in the fitting commands to print out fewer updates, since you will look at these graphically instead of as text

```python
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))


# Add the output layer
model_2.add(Dense(2, activation='softmax'))


# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
```
![121](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/121.png)

> #### Adding layers to a network
> > *  create a similar network with 3 hidden layers (still keeping 50 units in each layer)

```python
# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation='relu', input_shape=input_shape))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(50, activation='relu'))


# Add the output layer
model_2.add(Dense(2, activation='softmax'))


# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

```
![122](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/122.png)

### Thinking about model capacity

![123](https://github.com/htaiwan/note_data_scientist_with_python/blob/master/Asset/123.png)

### Stepping up to images
> #### Building your own digit recognition model
> > * already done the basic manipulation of the MNIST dataset
> > * X and y loaded and ready to model with
> > * 28 x 28 grid fla ened to 784 values for each image
> > * Value in each part of array denotes darkness of that pixel

```python
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.3)
```
