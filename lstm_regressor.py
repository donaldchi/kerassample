# coding: utf-8

import numpy as np
np.random.seed(2018)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006

import os
os.environ["KERAS_BACKEND"] = "theano"

def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (1*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[1,:], res[1,:], 'r', xs[1,:], seq[1,:], 'b--' )
    # plt.show()

    # use np.newaxis adding dimension
    return [seq[:,:,np.newaxis], res[:, :, np.newaxis], xs]

model = Sequential()

# build a LSTM RNN
model.add(LSTM(
    units=CELL_SIZE,
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
    return_sequences=True,  # output result per compute not just at the end of the compute
    stateful=True,  # if there is relationship between batch
))

# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))  # TimeDistributed : specify time series

adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)

print('Training ... ')
for step in range(501):
    X_batch, y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    plt.plot(xs[0,:], y_batch[0].flatten(), 'r', xs[0,:], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim(-1.2, 1.2)
    plt.draw()
    plt.pause(0.5)
    if step%10 == 0:
        print('train cost: ', cost)