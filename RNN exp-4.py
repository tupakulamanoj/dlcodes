import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

dataset, info = tfds.load('imdb_reviews',with_info =True,as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec
for example, label in train_dataset.take(1):
  print('text: ',example.numpy())
  print('label: ',label.numpy())
  BUFFER_SIZE =10000
  BATCH_SIZE = 64
  train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  VOCAB_SIZE = 1000
  encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
  encoder.adapt(train_dataset.map(lambda text, label: text))
  model = tf.keras.Sequential([encoder,
                               tf.keras.layers.Embedding(
                                   input_dim=len(encoder.get_vocabulary()),
                                   output_dim=64,

                                   mask_zero=True),
                               tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                               tf.keras.layers.Dense(64, activation='relu'),
                               tf.keras.layers.Dense(1)
                               ])
  
  sample_text = ('The movie was cool. The animation and the graphics were out of the world. I would rcommend this movie')
  prediction = model.predict(np.array([sample_text]))
  print(prediction[0])
  print("$$$$$$$$$$$$")
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])
  history = model.fit(train_dataset, epochs=2,
                      validation_data=test_dataset,
                      validation_steps=30)
  test_loss, test_acc = model.evaluate(test_dataset)
  print('Test Loss:',test_loss)
  print('Test Accuaracy:', test_acc)
