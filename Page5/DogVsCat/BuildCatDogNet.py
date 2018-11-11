from keras import models
from keras import layers
from keras.applications import VGG16


class CatDogNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=512, activation='relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.summary()
        return model

    @staticmethod
    def buildWithDropout(width, height, depth, classes):
        '''
        with dropout net
        :param width:
        :param height:
        :param depth:
        :param classes:
        :return: model
        '''
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))  # why set in here
        model.add(layers.Flatten())
        model.add(layers.Dense(units=512, activation='relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.summary()
        return model

    @staticmethod
    def buildDenseNet(dim):
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=dim))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    @staticmethod
    def BuildWithFreezeVGGNet(shape):
        vgg = VGG16(include_top=False,
                    weights='imagenet',
                    input_shape=shape)
        vgg.summary()
        vgg.trainable = False

        model = models.Sequential()
        model.add(vgg)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))

        model.summary()
        return model

    @staticmethod
    def BuildFintuneVGGNet(shape):
        '''
        fine tune pates of VGG net
        :param shape:
        :return:
        '''
        vgg = VGG16(include_top=False,
                    weights='imagenet',
                    input_shape=shape)
        vgg.trainable = True
        vgg.summary()
        set_trainable = False
        for layer in vgg.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True

            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        vgg.summary()

        model = models.Sequential()
        model.add(vgg)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        return model
