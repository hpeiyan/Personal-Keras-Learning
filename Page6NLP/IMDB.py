import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from Page6NLP.NLPNet import EmbbedingNet
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

texts = []
labels = []
# main_dir = r'/Users/zzc20160628-14/Downloads/aclImdb'
main_dir = r'/home/bigdata/Documents/Personal-Keras-Learning/data/aclImdb'
train_dir = os.path.join(main_dir, 'train')
test_dir = os.path.join(main_dir, 'test')
classes = ['neg', 'pos']
for index, i in enumerate(classes):
    # print(index)
    sub_path = os.path.join(train_dir, i)
    for path in os.listdir(sub_path):
        # print(path)
        with open(os.path.join(sub_path, path)) as f:
            texts.append(f.read())
            labels.append(index)
print('texts\' length: {}'.format(len(texts)))
print('labels\' length: {}'.format(len(texts)))

maxlen = 100
train_sample = 200
val_sample = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts=texts)
sequences = tokenizer.texts_to_sequences(texts=texts)
world_index = tokenizer.word_index
print('found {} unique word in texts'.format(len(world_index)))
data = pad_sequences(sequences=sequences, maxlen=maxlen)
labels = np.asarray(labels, dtype='float32')
print('the shape of data: {}'.format(data.shape))
print('the shape of label: {}'.format(labels.shape))
index_data = np.arange(data.shape[0])
np.random.shuffle(index_data)
data = data[index_data]
labels = labels[index_data]
print('after shuffle, the shape of data: {}'.format(data.shape))
print('after shuffle, the shape of label: {}'.format(labels.shape))
x_train = data[:train_sample]
y_train = labels[:train_sample]
x_val = data[train_sample:train_sample + val_sample]
y_val = labels[train_sample:train_sample + val_sample]
print('the shape of x_train: {}'.format(x_train.shape))
print('the shape of y_train: {}'.format(y_train.shape))
print('the shape of x_val: {}'.format(x_train.shape))
print('the shape of y_val: {}'.format(y_train.shape))

# vol_dir = r'/Users/zzc20160628-14/Downloads/glove.6B'
vol_dir = r'/home/bigdata/Documents/Personal-Keras-Learning/data/glove.6B'
embedding_index = {}
with open(os.path.join(vol_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coef
print('found {} unique word in glove.6b'.format(len(embedding_index)))

embedding_dim = 100
embedding_matrric = np.zeros(shape=(max_words, embedding_dim))

for word, i in world_index.items():
    if i < max_words:
        emb_vector = embedding_index.get(word)
        if emb_vector is not None:
            embedding_matrric[i] = emb_vector

model = EmbbedingNet.buildEmbNetWithPreTrain(max_words, embedding_dim, maxlen, embedding_matrric)
# model = EmbbedingNet.buildEmbNetWithoutPreTrain(max_words, embedding_dim, maxlen)
model.compile(optimizer=RMSprop(),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=50,
                    validation_data=(x_val, y_val))
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

model.save(r'./pre_train_glove_model.h5')
