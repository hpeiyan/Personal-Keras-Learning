from keras import models
from keras import layers


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
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, depth)))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))# why set in here
        model.add(layers.Flatten())
        model.add(layers.Dense(units=512, activation='relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.summary()
        return model

    @staticmethod
    def buildDenseNet(dim):
        model = models.Sequential()
        model.add(layers.Dense(256,activation='relu',input_dim=dim))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation='sigmoid'))
        return model