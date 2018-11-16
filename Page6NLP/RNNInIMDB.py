from keras.datasets import imdb
from Utils import INFO, plotHistory
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import models, layers

max_features = 10000
maxlen = 500
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
y_train = np.asarray(y_train, 'float32')
y_test = np.asarray(y_test, 'float32')

model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.SimpleRNN(units=32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=100,
                    validation_split=0.2)
plotHistory(history)

print(INFO() + 'End.')
