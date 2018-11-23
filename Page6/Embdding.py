from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing

from keras import models
from keras import layers

# embedding_layer = Embedding(1000, 64)

max_feature = 1000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = models.Sequential()
model.add(layers.Embedding(1000, 8, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

print('end')
