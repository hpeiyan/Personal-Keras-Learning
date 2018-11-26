from keras import models
from keras import layers

s_model = models.Sequential()
s_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
s_model.add(layers.Dense(32, activation='relu'))
s_model.add(layers.Dense(10, activation='softmax'))
s_model.summary()

in_tensor = layers.Input(shape=(64,),name='input_test')
x = layers.Dense(32, activation='relu')(in_tensor)
x = layers.Dense(32, activation='relu')(x)
out_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(in_tensor, out_tensor)
model.summary()
