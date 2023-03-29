import tensorflow as tf
import visualkeras

from tensorflow import keras

from keras.models import Sequential

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.models import Model

model = Sequential()

model.add(Conv2D(64, (4,4),input_shape=(32,32,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(4,4), input_shape=(32,32,3),activation='relu',padding='same' ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()
visualkeras.layered_view(model)
