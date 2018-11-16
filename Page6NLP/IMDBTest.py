import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
from Page6NLP.NLPNet import EmbbedingNet

test_main_dir = r'/Users/zzc20160628-14/Downloads/aclImdb/test'
result_types = ['neg', 'pos']
results = []
labels = []
max_words = 10000
maxlen = 100
embedding_dim = 100

for index, type in enumerate(result_types):
    test_dir = os.path.join(test_main_dir, type)
    for file in os.listdir(test_dir):
        if file[-4:] == '.txt':
            with open(os.path.join(test_dir, file)) as f:
                results.append(f.read())
                labels.append(index)

print('the length of results: {}'.format(len(results)))
print('the length of labels: {}'.format(len(labels)))

tokenizer = Tokenizer(num_words=max_words)
sequence = tokenizer.texts_to_sequences(texts=results)
x_test = pad_sequences(sequences=sequence, maxlen=maxlen)
y_test = np.asarray(labels, dtype='float32')

print('the shape of test x_test: {}'.format(x_test.shape))
print('the shape of y_test: {}'.format(y_test.shape))

model = EmbbedingNet.buildEmbNetWithoutPreTrain(max_words, embedding_dim, maxlen)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.load_weights(r'./pre_train_glove_model.h5')

test_loss, test_acc = model.evaluate(x_test, y_test)
print('tes mode of acc: {}'.format(test_acc))
