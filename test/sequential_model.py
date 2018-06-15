#Use Keras library with Tensorflow backend
import keras

#Use Sequential model with Dense layer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#Create output array as 3 times input with some randomness
trX = np.linspace(-1, 1, 101)
trY = 3 * trX + np.random.randn(*trX.shape) * 0.33

#Set model and add first layer
model=Sequential()
model.add(Dense(input_dim=1, units=1, kernel_initializer='uniform', activation='linear'))

#Can check what weight and bias the model has been initialised with
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Initial weight and bias: %.2f, %.2f'%(w_init, b_init))

#Use simple gradient descent as optimiser, and calculate loss as mean squared error
model.compile(optimizer='sgd', loss='mse')

#Feed in data
#Set number of epochs (number of times the network will iterate over the whole dataset)
#Note: nb_epoch has been renamed epochs from Keras 2.0 onwards
model.fit(trX, trY, epochs=200, verbose=1)

#Check what w and b have been updated to
#w should be close to 3
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Final weight and bias: %.6f, %.6f'%(w_init, b_init))

#Can save the w and b values using h5py package
#Simply load these values and resume training using model.load_weights("sequential_model.py")
model.save_weights("sequential_model.h5")