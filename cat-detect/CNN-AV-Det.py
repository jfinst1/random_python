import os
import cv2 as cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

#   Waiting for the break of day...
#   Searching for something to say...

img_width, img_height, img_num_channels = 64, 64, 3

#   Flashing lights against the sky...
#   Giving up I close my eyes...

def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(label)
    return images, labels

#   Sitting cross-legged on the floor..

real_images, real_labels = load_images('real_directory', 0)
fake_images, fake_labels = load_images('fake_directory', 1)

#   25 or 6 to 4!!

data = np.array(real_images + fake_images)
labels = np.array(real_labels + fake_labels)

data = data.astype('float32') / 255.0  

train_images, test_images, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, img_num_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)