# coding: utf-8

import numpy as np
np.random.seed(2018)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# X_train : (60000*28*28), Y_train:(60000), X_test: (60000*28*28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # changed to array(60000*784)
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_train = np_utils.to_categorical(y_train)  # convert int to vector
y_test = np_utils.to_categorical(y_test)  # convert int to vector

# build model
model = Sequential([
    Dense(output_dim=32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
rmsporp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(
    optimizer=rmsporp,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print('Training ... ')
model.fit(X_train, y_train, nb_epoch=100, batch_size=32)

print('Testing ... ')
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
