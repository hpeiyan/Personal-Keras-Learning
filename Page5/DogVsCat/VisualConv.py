from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models

model = load_model(r'./model.h5')
model.summary()

imagePath = r'/Users/zzc20160628-14/Documents/Personal-Keras-Learning/Page5/DogVsCat/cat_and_dog_small/train/cat/cat.0.jpg'
img = image.load_img(imagePath, target_size=(150, 150))
image_tensor = image.img_to_array(img=img)
image_tensor = np.expand_dims(image_tensor, axis=0)
image_tensor /= 255.0
print('image_tensor.shape: ' + str(image_tensor.shape))

# plt.imshow(image_tensor[0])
# plt.show()

output_layers = [layer.output for layer in model.layers[:8]]
print(output_layers[0])

activation_model = models.Model(inputs=model.input, outputs=output_layers)
activation = activation_model.predict(image_tensor)
plt.imshow(activation[0][0, :, :, 15], cmap='summer')
plt.show()

print('end')
