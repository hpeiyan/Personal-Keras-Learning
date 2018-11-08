from Page5.DogVsCat.BuildCatDogNet import CatDogNet
from Page5.DogVsCat.ProcessData import PreProcessData
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir, val_dir, test_dir = PreProcessData.process()

model = CatDogNet.build(150, 150, 3, 1)
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1 / 255.0)
train_gen = train_datagen.flow_from_directory(directory=train_dir, target_size=(150, 150), batch_size=20,
                                              class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_gen = val_datagen.flow_from_directory(directory=val_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1 / 255.0)
test_gen = test_datagen.flow_from_directory(directory=test_dir, target_size=(150, 150), batch_size=20,
                                            class_mode='binary')

history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=val_gen,
                              validation_steps=50)
modelSavePath = r'./model.h5'
model.save(modelSavePath)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(1, len(acc) + 1)
plt.plot(epoch, acc, 'bo', label='Training accuracy')
plt.plot(epoch, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation data\'s accuracy')
plt.legend()

plt.figure()
plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation data\'s loss')
plt.legend()

plt.show()
