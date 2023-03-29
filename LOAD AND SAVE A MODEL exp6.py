import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Activation 
import matplotlib.pyplot as plt 

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Cast the records into float values
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')

# Normalize image pixel values by dividing by 255
gray_scale = 255 
x_train /= gray_scale 
x_test /= gray_scale 

# Define the Sequential model
model = Sequential([
  # Reshape data to 28*28 rows
  Flatten(input_shape=(28, 28)),
  # Dense layer 1
  Dense(256, activation='sigmoid'),
  # Dense layer 2
  Dense(128, activation='sigmoid'),
  # Output layer
  Dense(10, activation='sigmoid'),
])

# Print the model summary
model.summary()

# Saving and loading the .h5 model

# Save the model
model.save_weights('MLPWeights.h5')
print('Model Saved!')

# Load the saved model
savedModel = model.load_weights('MLPWeights.h5')
print('Model Loaded!')
