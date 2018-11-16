from keras import layers
from keras import models

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1, 1, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 60,554
Trainable params: 60,554
Non-trainable params: 0
_________________________________________________________________
'''


class FirstKerasNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=classes, activation='softmax'))
        model.summary()
        return model
