import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

def sigmoid(z):
  return 1/(1 + np.exp(-z))

# Produce a neural network randomly initialized
def initialize_parameters(n_x, n_h, n_y):
  W1 = np.random.randn(n_h, n_x)
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h)
  b2 = np.zeros((n_y, 1))

  parameters = {
  "W1": W1,
  "b1" : b1,
  "W2": W2,
  "b2" : b2
  }
  print("W1", parameters["W1"].shape)
  print("b1", parameters["b1"].shape)
  print("W2", parameters["W2"].shape)
  print("b2", parameters["b2"].shape)
  return parameters

# Evaluate the neural network
def forward_prop(X, parameters):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  # Z value for Layer 1
  Z1 = np.dot(W1, X) + b1
  # Activation value for Layer 1
  A1 = np.tanh(Z1)
  # Z value for Layer 2
  Z2 = np.dot(W2, A1) + b2
  # Activation value for Layer 2
  A2 = sigmoid(Z2)

  cache = {
    "A1": A1,
    "A2": A2
  }

  return A2, cache

# Evaluate the error (i.e., cost) between the prediction made in A2 and the provided labels Y 
# We use the Mean Square Error cost function
def calculate_cost(A2, Y):
  # m is the number of examples
  cost = np.sum(((A2 - Y) ** 2).mean(axis=1))/m
  return cost

def normalization(x, min_x, max_x, nh, nl):
    return (x - min_x) * (nh - nl) / (max_x-min_x) + nl
    

# Apply the backpropagation
def backward_prop(X, Y, cache, parameters):
  A1 = cache["A1"]
  A2 = cache["A2"]

  W2 = parameters["W2"]

  # Compute the difference between the predicted value and the real values
  dZ2 = A2 - Y
  dW2 = np.dot(dZ2, A1.T)/m
  db2 = np.sum(dZ2, axis=1, keepdims=True)/m
  # Because d/dx tanh(x) = 1 - tanh^2(x)
  dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
  dW1 = np.dot(dZ1, X.T)/m
  db1 = np.sum(dZ1, axis=1, keepdims=True)/m

  grads = {
    "dW1": dW1,
    "db1": db1,
    "dW2": dW2,
    "db2": db2
  }

  return grads

# Third phase of the learning algorithm: update the weights and bias
def update_parameters(parameters, grads, learning_rate):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]

  dW1 = grads["dW1"]
  db1 = grads["db1"]
  dW2 = grads["dW2"]
  db2 = grads["db2"]

  W1 = W1 - learning_rate*dW1
  b1 = b1 - learning_rate*db1
  W2 = W2 - learning_rate*dW2
  b2 = b2 - learning_rate*db2
  
  new_parameters = {
    "W1": W1,
    "W2": W2,
    "b1" : b1,
    "b2" : b2
  }

  return new_parameters

# model is the main function to train a model
# X: is the set of training inputs
# Y: is the set of training outputs
# n_x: number of inputs (this value impacts how X is shaped)
# n_h: number of neurons in the hidden layer
# n_y: number of neurons in the output layer (this value impacts how Y is shaped)
def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
  parameters = initialize_parameters(n_x, n_h, n_y)
  for i in range(0, num_of_iters+1):
    a2, cache = forward_prop(X, parameters)
    cost = calculate_cost(a2, Y)
    grads = backward_prop(X, Y, cache, parameters)
    parameters = update_parameters(parameters, grads, learning_rate)
    if(i%100 == 0):
      print('Cost after iteration# {:d}: {:f}'.format(i, cost))
      ITERATIONS.append(i)
      COSTS.append(cost)

  return parameters

# Make a prediction
# X: represents the inputs
# parameters: represents a model
# the result is the prediction
def predict(X, parameters):
  a2, cache = forward_prop(X, parameters)
  yhat = a2
  yhat = np.squeeze(yhat)
  # We will return the index of the neuron with the highest value
  return getMaxOutputIndex(yhat)
  

def getMaxOutputIndex(values) -> int:
  """
  Given an array of numerical values, it returns the index where the highest value is found.

  Example
  keepMaxOutputValue([0.97144037, 0.87981931, 0.893457, 0.9423538 , 0.94605132, 0.97859269])
  ---> 5
  """
  index = 0
  maxValue = 0
  for i, value in enumerate(values):
    if value > maxValue:
      index = i
      maxValue = value
  return index
  



#One hot encoding
# The goal of the neural network is to predict the type of the star based on the rest 
# of the data in the dataset. 
# 
# The Type of the star corresponds to the fifth column of the dataset.

# COLUMN 5 - OUTPUT
encoding = {
  0: [1,0,0,0,0,0],
  1: [0,1,0,0,0,0],
  2: [0,0,1,0,0,0],
  3: [0,0,0,1,0,0],
  4: [0,0,0,0,1,0],
  5: [0,0,0,0,0,1],
}

# COLUMN 6 - Star color
color_encoding = {
  "Red": 0,
  "Blue white": 1,
  "Blue white ": 1,
  "Blue White": 1,
  "Blue-white": 1,
  "Blue-White": 1,
  "White": 2,
  "white": 2,
  "Yellowish White" : 3,
  "yellow-white": 3,
  "White-Yellow": 3,
  "Pale yellow orange" : 4,
  "Whitish": 5,
  "Orange": 6,
  "yellowish": 7,
  "Yellowish": 8,
  "Orange-Red": 9,
  "Blue": 10,
  "Blue ": 10
}

# COLUMN 7 - Spectral Class
spectral_class_encoding = {
  "M": 0,
  "B": 1,
  "A": 2,
  "F": 3,
  "O": 4,
  "K": 5,
  "G": 6,
}

##################### Getting the data

# Init plot values
plt.grid(zorder=0)
ITERATIONS = []
COSTS = []

# Init input and output arrays from csv
input_data = np.array([])
output_data = np.array([], dtype=int)

# Reading csv

data = pd.read_csv("6-star.csv", sep=",")

#the data is disordered so that later we can extract random data in the tests and training
data=data.iloc[np.random.permutation(len(data))].reset_index(drop=True)

#the input and output arrays are made from the dataset
input=np.ones((6,240), dtype=float)
output_data=np.zeros((1,240),dtype=int)

for i in range(240):
  input[0][i]=normalization(float(data['Temperature (K)'][i]), 1939, 40000, 1, 0)
  input[1][i]=normalization(float(data['Luminosity(L/Lo)'][i]), 0.00008, 849420, 1, 0)
  input[2][i]=normalization(float(data['Radius(R/Ro)'][i]), 0.0084, 1948.5, 1, 0)
  input[3][i]=normalization(float(data['Absolute magnitude(Mv)'][i]), -11.92, 20.06, 1, 0)
  input[4][i]=normalization(float(color_encoding[data['Star color'][i]]), 0, 10, 1, 0)
  input[5][i]=normalization(float(spectral_class_encoding[data['Spectral Class'][i]]), 0, 6, 1, 0)

  output_data[0][i] = int(data['Star type'][i])


#We will train the network with 80% of the data and test with the other 20%
train_number = int(0.8*240)
test_number = int(0.2*240)

train_input=np.zeros((6,train_number), dtype=float)
test_input=np.zeros((6,test_number), dtype=float)

test_output = np.array([], dtype=int)
train_output = np.array([], dtype=int)

#The examples are separated into different matrices for testing and training
for i in range(240):
  if i < train_number:
    for j in range(6):
      train_input[j][i] =input[j][i]
    train_output = np.append(train_output, np.array([encoding[output_data[0][i]]]))

  else:
    for j in range(6):
      test_input[j][i-train_number] = input[j][i]
    test_output = np.append(test_output, np.array([encoding[output_data[0][i]]]))

# We must to re-order the data
train_output = train_output.reshape(train_number, 6)
test_output = test_output.reshape(test_number, 6)


# Define a model
train_X = train_input
train_Y = train_output.T
n_x = 6
n_h = 6
n_y = 6
m = train_X.shape[1]
number_of_iterations = 10000
learning_rate = 0.5

test_X = test_input
test_Y = test_output


# Training our neural network
trained_parameters = model(train_X, train_Y, n_x, n_h, n_y, number_of_iterations, learning_rate)

# Init confusion matrix with zeros, we have 6 classes to predict.
confusion_matrix=np.zeros((6,6))

# For each example of the test data the predictions are obtained, we store the results in the confusion matrix
for i, starIndex in enumerate(range(test_number)):
  predictedValue = predict(np.array([
      [test_X[0][starIndex]],
      [test_X[1][starIndex]],
      [test_X[2][starIndex]],
      [test_X[3][starIndex]],
      [test_X[4][starIndex]],
      [test_X[5][starIndex]],
    ]), trained_parameters)
  expectedValue = int(np.where(test_Y[starIndex] == 1)[0])
  confusion_matrix[predictedValue][expectedValue] += 1

def precision(label):
  """
  Calculates the precision of the predictions of our neural network. 
  The label corresponds to the star type.
  """
  total_predicted = np.apply_along_axis(sum, 1, confusion_matrix)[label]
  true_positives = confusion_matrix[label][label]
  
  return true_positives/total_predicted

def recall(label):
  """
  Calculates the recall of the predictions of our neural network.
  """
  total_GoldLabel = np.apply_along_axis(sum, 0, confusion_matrix)[label]
  true_positives = confusion_matrix[label][label]
  
  return true_positives/total_GoldLabel 

print("\nConfusion Matrix:")
print(confusion_matrix)
print()

for label in range(6):
  print("Star Type %d - Precision: %s %% | Recall: %s %%" % (label, precision(label)*100, recall(label)*100))

##### CHART: COST vs ITERATIONS
plt.plot(ITERATIONS, COSTS, color="blue")
plt.title("Error measurement predicting a star. (6 neurons in hidden layer)")
plt.ylabel("MSE (Mean Squared Error)")
plt.xlabel("Number of Iterations")
plt.show()
plt.close()
