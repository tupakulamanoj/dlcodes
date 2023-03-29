import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers


bc = datasets.load_breast_cancer()
x = bc.data
y = bc.target


network = models.Sequential()
network.add(layers.Dense(32,activation='relu', input_shape=(30,)))
network.add(layers.Dense(32,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))


network.compile(optimizer=optimizers.RMSprop(lr=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3 ,stratify=y, random_state=42)

history = network.fit(x_train,y_train,
                      validation_data=(x_test, y_test),
                      epochs=10,
                      batch_size=20)


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['accuracy']

val_accuracy = history_dict['val_accuracy']

epochs = range(1, len(loss_values) +1)
fig, ax = plt.subplots(1,2, figsize=(14,6))



ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
ax[0].plot(epochs, val_accuracy, 'b', label='validation accuracy')
ax[0].set_title('Training & validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuarcy', fontsize=16)
ax[0].legend()

ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='validation accuracy')
ax[1].set_title('Training & validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()
