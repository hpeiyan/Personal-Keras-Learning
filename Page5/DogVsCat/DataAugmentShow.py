import os
from keras.preprocessing import image
import matplotlib.pyplot as plt

INFO = '[INFO] '

train_cat_dir = r'./cat_and_dog_small/train/cat'
fnames = [os.path.join(train_cat_dir, fname) for fname in os.listdir(train_cat_dir)]
print(str(os.path.exists(fnames[0])))
print(INFO + fnames[6])

img = image.load_img(path=fnames[6], target_size=(150, 150))
img_arr = image.img_to_array(img=img)
img_arr = img_arr.reshape((1,) + img_arr.shape)

datagen = image.ImageDataGenerator(rescale=1 / 255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
i = 0
for batch in datagen.flow(img_arr, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 6 == 0:
        break

plt.show()
