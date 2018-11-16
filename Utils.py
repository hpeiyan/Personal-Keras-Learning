import os
import matplotlib.pyplot as plt


def ifNoneCreateDirs(filePath):
    if not os.path.exists(filePath):
        os.makedirs(filePath)


def INFO():
    return '[INFO] '


def plotHistory(history):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'bo', label='train_acc')
    plt.plot(epochs, val_acc, 'b', label='val_acc')
    plt.legend()
    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='train_loss')
    plt.plot(epochs, val_loss, 'b', label='val_loss')
    plt.legend()
    plt.show()
