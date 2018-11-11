from Page5.DogVsCat.BuildCatDogNet import CatDogNet
import os
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = r'./cat_and_dog_small'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

model = CatDogNet.BuildFintuneVGGNet((150, 150, 3))
data_gen = ImageDataGenerator(rescale=1. / 255)
train_gen = data_gen.flow_from_directory(directory=train_dir,
                                         target_size=(150, 150),
                                         class_mode='binary',
                                         batch_size=20)
val_gen = data_gen.flow_from_directory(directory=val_dir,
                                       target_size=(150, 150),
                                       class_mode='binary',
                                       batch_size=20)
model.compile(optimizer=RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=50,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=50)
model.save('./ClassifyWithFreezeNet.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, labels='train acc')
plt.plot(epochs, val_acc, labels='val acc')
plt.legend()

plt.figure()
plt.plot(epochs, loss, labels='train loss')
plt.plot(epochs, val_loss, labels='val loss')
plt.legend()

plt.show()
