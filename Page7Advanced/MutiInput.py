from keras import models
from keras import layers
from keras.optimizers import RMSprop
import numpy as np

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = layers.Input(shape=(None,), name='text', dtype='int32')
embed_text = layers.Embedding(64, text_vocabulary_size)(text_input)
encoded_text = layers.LSTM(32)(embed_text)

question_input = layers.Input(shape=(None,), name='question', dtype='int32')
embed_question = layers.Embedding(input_dim=64, output_dim=question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(32)(embed_question)

concatenated = layers.concatenate([encoded_text, encoded_question])

output_tensor = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = models.Model([text_input, question_input], output_tensor)
model.compile(optimizer=RMSprop(),
              loss='categorical_crossentropy',
              metrics=['acc'])

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answer = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

model.fit([text, question], answer, epochs=10, batch_size=128)
