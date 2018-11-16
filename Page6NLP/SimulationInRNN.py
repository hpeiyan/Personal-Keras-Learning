import numpy as np

timesteps = 100
input_feature = 32
output_feature = 64

inputs = np.random.random(size=(timesteps, input_feature))
state_t = np.zeros(shape=(output_feature,))

W = np.random.random(size=(output_feature, input_feature))
U = np.random.random(size=(output_feature, output_feature))
b = np.random.random(size=(output_feature,))

successive_outputs = []

for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

print('the content of successive_outputs: {}'.format(successive_outputs))
final_output_sequence = np.concatenate(successive_outputs, axis=0)
print('the content of final_output_sequence: {}'.format(final_output_sequence))
