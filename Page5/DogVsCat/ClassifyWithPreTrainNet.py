from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import sys
sys.path.append('../..')
from Utils import INFO
from BuildCatDogNet import CatDogNet
from keras import optimizers
import matplotlib.pyplot as plt

conv_base = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(150, 150, 3))

conv_base.summary()

base_dir = r'./cat_and_dog_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

data_gen = ImageDataGenerator(rescale=1 / 255.0)
batch_size = 1


def extract_feature(directory, sample_count):
    '''
    提取特征
    :param directory:
    :param sample_count:
    :return:
    '''
    feature = np.zeros(shape=(sample_count, 4, 4, 512))
    # block5_pool (MaxPooling2D)   (None, 4, 4, 512)
    label = np.zeros(shape=(sample_count))
    generator = data_gen.flow_from_directory(directory=directory,
                                             target_size=(150, 150),
                                             batch_size=batch_size,
                                             class_mode='binary')  # means 二分类
    i = 0
    for input_batch, label_batch in generator:
        feature_batch = conv_base.predict(x=input_batch)
        feature[i * batch_size:(i + 1) * batch_size] = feature_batch
        label[i * batch_size:(i + 1) * batch_size] = label_batch
        i += 1
        print(INFO() + directory + ' {}\n'.format(i))
        if i * batch_size >= sample_count:
            break

    return feature, label


print(INFO() + 'start extract feature...')
train_fea, train_lab = extract_feature(directory=train_dir, sample_count=200)
val_fea, val_lab = extract_feature(directory=validation_dir, sample_count=100)
test_fea, test_lab = extract_feature(directory=test_dir, sample_count=100)
print(INFO() + 'end extract feature...')

train_fea = np.reshape(train_fea, (200, 4 * 4 * 512))
val_fea = np.reshape(val_fea, (100, 4 * 4 * 512))
test_fea = np.reshape(test_fea, (100, 4 * 4 * 512))

model = CatDogNet.buildDenseNet(4 * 4 * 512)
model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x=train_fea,
                    y=train_lab,
                    epochs=20,
                    validation_data=(val_fea, val_lab))
acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='train_acc')
plt.plot(epochs, val_acc, 'b', label='val_acc')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='train_loss')
plt.plot(epochs, val_loss, 'b', label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

print(INFO() + 'Done')
