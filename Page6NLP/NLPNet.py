from keras import models
from keras import layers


class EmbbedingNet:
    @staticmethod
    def buildEmbnet(input_dim, output_dim, input_length):
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        return model
