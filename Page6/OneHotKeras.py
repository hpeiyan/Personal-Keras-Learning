from keras.preprocessing.text import Tokenizer
import numpy as np

samples = ['one cat sat on the mat', 'one dog ate my homework']


def worldOneHotWithKeras():
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(samples)
    sequence = tokenizer.texts_to_sequences(samples)
    one_hot_result = tokenizer.texts_to_matrix(samples, 'binary')
    word_index = tokenizer.word_index
    print('found %s unique token.' % len(word_index))


def worldOneHotWithBreak():
    dimentionality = 1000
    max_length = 10
    results = np.zeros(shape=(len(samples),
                              max_length,
                              dimentionality))
    for i, sample in enumerate(samples):
        for j, world in list(enumerate(sample.split()))[:max_length]:
            index = abs(hash(world)) % dimentionality
            results[i, j, index] = 1

    print(results.shape)


worldOneHotWithBreak()
