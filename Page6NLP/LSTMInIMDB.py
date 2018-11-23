from keras.datasets import imdb
from keras import models
from keras import layers
from Utils import plotHistory, INFO
from keras.preprocessing.sequence import pad_sequences

max_feature = 10000
max_len = 500

print(INFO() + 'loading data')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
model = models.Sequential()
model.add(layers.Embedding(max_feature, 32))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
print(INFO() + 'training')
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
x_train = pad_sequences(x_train,maxlen=max_len)
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=10,
                    validation_split=0.2)
plotHistory(history)
print(INFO() + 'end')
