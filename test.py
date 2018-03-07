# coding: utf-8
import numpy as np
np.random.seed(2018)

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
X_train, y_train = X[:160], Y[:160]
X_test, y_test = X[160:], Y[160:]

model = Sequential()
model.add(Dense(input_shape=(1,), units=1))
model.compile(loss='mse', optimizer='sgd')

for step in range(301):
    cost = model.train_on_batch(X_train, y_train)

# save model
print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')  # HDF5
del model

# load model
model = load_model('my_model.h5')
print('test after save: ', model.predict(X_test[0:2]))

# save weights
# model.save_weights('my_model_weights.h5')
# model.load_weights('my_model_weights.h5')

# only save model structure
# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)