import os
from Utils import INFOWithResult, plotHistory
import numpy as np
import matplotlib.pyplot as plt
from NLPNet import EmbbedingNet

INFOWithResult('loading data')
main_dir = r'/home/bigdata/Documents/Personal-Keras-Learning/data/jena_climate'
climate_dir = os.path.join(main_dir, 'jena_climate_2009_2016.csv')

lines = []
headers = []
with open(climate_dir) as f:
    content = f.read()
    lines = content.split('\n')
    headers = lines[0].split(',')
    lines = lines[1:]

INFOWithResult(headers)
INFOWithResult(len(lines))

float_data = np.zeros(shape=(len(lines), len(headers) - 1))
INFOWithResult(float_data.shape)

for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

INFOWithResult(float_data[:5, :])
temp = float_data[:, 2]
INFOWithResult(temp)
# plt.plot(range(1, len(temp) + 1), temp, label='temp')
# plt.legend()
# plt.figure()
# plt.plot(range(0, 1400), temp[:1400], label='1400 temp')
# plt.legend()
# plt.show()

## 标准化处理

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

INFOWithResult(float_data[:5, :])


def generator(data, loockback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1

    i = min_index + loockback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + loockback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + loockback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros(shape=(len(rows),
                                  loockback // step,
                                  data.shape[-1]))
        targets = np.zeros(len(rows), )
        for j, row in enumerate(rows):
            indices = range(rows[j] - loockback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train = []
label = []

train_gen = generator(data=float_data,
                      loockback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(data=float_data,
                    loockback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(data=float_data,
                     loockback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

INFOWithResult('launch generator')
# for sample, label in train_gen:
#     INFOWithResult(sample)
#     INFOWithResult(label)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


# evaluate_naive_method()

model = EmbbedingNet.buildBasicMLNet(shape=(lookback // step, float_data.shape[-1]))
model.compile(optimizer='rmsprop',
              loss='mae')
history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=50,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

plotHistory(history)
