from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Embedding(input_dim=10000, output_dim=32))
model.add(layers.SimpleRNN(units=32,return_sequences=True))
model.add(layers.SimpleRNN(units=32,return_sequences=True))
model.add(layers.SimpleRNN(units=32,return_sequences=True))
model.add(layers.SimpleRNN(units=32))
model.summary()
