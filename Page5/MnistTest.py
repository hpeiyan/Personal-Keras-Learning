from keras.datasets import mnist
from keras.utils import to_categorical
from Page5.FirstNet import FirstKerasNet

INFO = '[INFO] '

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = FirstKerasNet.build(28, 28, 1, 10)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(INFO + 'test_loss:{}\ntest_acc:{}'.format(test_loss, test_acc))
print(INFO + 'end')
