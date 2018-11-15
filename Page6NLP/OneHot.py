import numpy as np

import string


def worldConvert():
    global samples, token_index, sample, max_length, results, i, j, index
    samples = ['the cat sat on the mat.', 'the dog ate my homework']
    token_index = {}
    for sample in samples:
        for world in sample.split():
            if world not in token_index:
                token_index[world] = len(token_index) + 1
    max_length = 10
    results = np.zeros(shape=(len(samples),
                              max_length,
                              max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, world in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(world)
            results[i, j, index] = 1.
    print(results.shape)


def charConvert():
    global samples, token_index, max_length, results, i, sample, j, index
    samples = ['the cat sat on the mat.', 'the dog ate my homework']
    charecters = string.printable
    token_index = dict(zip(range(1, len(charecters) + 1), charecters))
    max_length = 50
    results = np.zeros(shape=(len(samples),
                              max_length,
                              max(token_index.keys()) + 1))
    for i, sample in enumerate(samples):
        for j, char in enumerate(sample):
            index = token_index.get(char)
            results[i,
                    j,
                    index] = 1
    print(results.shape)


if __name__ == '__main__':
    charConvert()