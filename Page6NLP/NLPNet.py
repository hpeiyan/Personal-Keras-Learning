from keras import models
from keras import layers


class EmbbedingNet:
    @staticmethod
    def buildEmbNetWithPreTrain(input_dim, output_dim, input_length, embedding_matrric):
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.layers[0].set_weights([embedding_matrric])
        model.layers[0].trainable = False
        model.summary()
        return model

    @staticmethod
    def buildEmbNetWithoutPreTrain(input_dim, output_dim, input_length):
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        return model

    @staticmethod
    def buildBasicMLNet(shape):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=shape))
        model.add(layers.Dense(32,activation='relu'))
        model.add(layers.Dense(1))
        model.summary()
        return model