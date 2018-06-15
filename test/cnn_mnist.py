#Note: Still don't understand convolution filters used in CNN

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
 
#Load MNIST data into train and test sets
#Note: Data already shuffled
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train, X_test have shape of (60000, 28, 28)
#Images are stored as [height, width, channels] in TF
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#Change to floats and normalise to [0,1] range
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Y_train, Y_test have shape of (60000,), i.e. just the label
#Change them to shape (60000, 10), with the label having value 1.0
#and the other 9 values just being 0.0
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#Use sequential model
model = Sequential()
 
#Set input layer as 2D convolution
#32 output filters in the convolution, using a 3x3 window
#Note: READ UP ON WHY CONVOLUTION FILTERS ARE USED(??)
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
#Add in another layer
model.add(Convolution2D(32, (3, 3), activation='relu'))
#Add in 2x2 MaxPooling window that slides a 2x2 window over layer 
#and takes max of the 4 values --> reduces parameters for model
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
#Weights need to be made one-dimensional before passing into Dense
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
#Select the loss function and optimizer
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fit model, iterating 10 times over dataset
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

model.save_weights("cnn_mnist.h5")