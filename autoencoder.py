# coding: utf-8
import numpy as np
np.random.seed(2018)

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# load data, x_train: (60000, 28, 28), x_test: (10000, 28, 28)
# y_test: (10000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# data pre-processing x: (60000, 28, 28) y: (60000, )
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized, range: [-0.5, 0.5]
x_test = x_test.astype('float32') / 255. - 0.5    # minmax_normalized, range: [-0.5, 0.5]
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# output dimension
encoding_dim = 2

# input placeholder
input_img = Input(shape=(784,))

# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# construct the auto encoder model
autoencoder = Model(inputs=input_img, outputs=decoded)

# construct the encoder model fro plotting
encoder = Model(inputs=input_img, outputs=encoder_output)

# compile auto encoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
print(encoded_imgs.shape)
plt.scatter(encoded_imgs[:,0], encoded_imgs[:,1], c=y_test)
plt.show()
