pip install tensorflow-macos==2.9.0
pip install tensorflow-metal==0.5.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

from keras.preprocessing.image import ImageDataGenerator
class_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
data_generator = ImageDataGenerator(rescale=1./255,validation_split=0.3)

train = data_generator.flow_from_directory("/Users/prabalkalhans/Desktop/Dog_Pictures/Images",target_size = (128,128),batch_size=20,class_mode='categorical',subset="training")

val = data_generator.flow_from_directory("/Users/prabalkalhans/Desktop/Dog_Pictures/Images",target_size = (128,128),batch_size=20,class_mode='categorical',subset="validation")

x_train, y_train = train.next()
x_val, y_val = val.next()

print(x_train[2].shape)
plt.imshow(x_train[2])
plt.show()

tf.keras.backend.clear_session()

model = Sequential([Conv2D(64, 7, activation="relu", padding="same",input_shape=(128,128,3)),MaxPool2D(2),Conv2D(128, 5, activation="relu", padding="same"),MaxPool2D(2),Conv2D(128, 3, activation="relu", padding="same"),MaxPool2D(2),Flatten(),Dropout(0.5),Dense(128, activation="relu"),Dropout(0.3),Dense(64, activation="relu"),Dense(4, activation="softmax")])

model.layers[0].trainable = False

model.compile(
    loss="categorical_crossentropy", 
    optimizer=Adam(learning_rate=1e-5), 
    metrics=["accuracy"])


hist = model.fit_generator(train,epochs=30,validation_data=val,)
print(predictions)


#This can be used for a 2 Class model, for e.g. "Happy and Sad"
train_2c = data_generator.flow_from_directory(
"/Users/prabalkalhans/Desktop/Dog_Pictures/Images",
target_size = (128,128),
batch_size=20,
class_mode='binary',
subset="training"
)

val_2c = data_generator.flow_from_directory(
"//Users/prabalkalhans/Desktop/Dog_Pictures/Images",
target_size = (128,128),
batch_size=20,
class_mode='binary',
subset="validation"
)

