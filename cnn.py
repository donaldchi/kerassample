import numpy as np
np.random.seed(2018)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

# X_train : (60000*28*28), Y_train:(60000), X_test: (60000*28*28)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# pre-processing
X_train = X_train.reshape(-1, 28, 28, 1)  # changed to array(60000*1*28*28)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train = np_utils.to_categorical(y_train, num_classes=10)  # convert int to vector
y_test = np_utils.to_categorical(y_test, num_classes=10)  # convert int to vector

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)
# build model
model = Sequential()
# 3*3のスライドwindow(filter?)が32個ある
model.add(Conv2D(
    filters=32,
    input_shape=(28, 28, 1),   # height * width*channels, channels comes last in tensorflow backend
    kernel_size=(3,3),
    activation='relu'
))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print('Training ... ')
model.fit(X_train, y_train, epochs=1, batch_size=32)

print('Testing ... ')
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
