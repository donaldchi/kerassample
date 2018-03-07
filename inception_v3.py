# coding: utf-8

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.image as mpimg
from scipy.misc import imresize
import numpy as np
import keras.backend as K
import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

K.clear_session()
img_size = 299

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# extend data set
train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=[.8, 1],
    channel_shift_range=30,
    fill_mode='reflect'
)

test_datagen = ImageDataGenerator()

# load images
def load_images(root, nb_img):
    all_imgs = []
    all_classes = []

    # read and resize dog img
    for i in range(nb_img):
        img_name = "{}/dog.{}.jpg".format(root, i+1)
        img_arr = mpimg.imread(img_name)
        resize_img_arr = imresize(img_arr, (img_size, img_size))
        all_imgs.append(resize_img_arr)
        all_classes.append(0)

    # read and resize cat img
    for i in range(nb_img):
        img_name = "{}/cat.{}.jpg".format(root, i+1)
        img_arr = mpimg.imread(img_name)
        resize_img_arr = imresize(img_arr, (img_size, img_size))
        all_imgs.append(resize_img_arr)
        all_classes.append(1)

    return np.array(all_imgs), np.array(all_classes)

X_train, y_train = load_images('./data/train', 1000)
X_test, y_test = load_images('./data/train', 400)
train_generator = train_datagen.flow(X_train, y_train, batch_size=64, seed = 13)
test_generator = test_datagen.flow(X_test, y_test, batch_size=64, seed = 13)

# read Inception v3 model
# include_top: if use dense layer at the end of the network model
# weights: if use weights pre-learnt by imagenet
base_model = InceptionV3(weights='imagenet', include_top=False)

# set the last layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid', kernel_regularizer=l2(.0005))(x)
model = Model(inputs=base_model.input, outputs=predictions)

# base_modelはweightsを更新しない
for layer in base_model.layers:
    layer.trainable = False

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# define callback function to save result
checkpointer = ModelCheckpoint(filepath='model.{epcho:02d}--{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=5, min_lr=0.001)

history = model.fit_generator(train_generator,
                    steps_per_epoch=2000,
                    epochs=10,
                    validation_data=test_generator,
                    validation_steps=800,
                    verbose=1,
                    use_multiprocessing=True,
                    callbacks=[reduce_lr, csv_logger, checkpointer])
