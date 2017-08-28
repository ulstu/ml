from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential,Model
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense,Dropout,Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# import h5py
from PIL import Image

import numpy as np
import os



data = []
labels = []

directory="sample"
figure=os.listdir(directory)

for name in figure:
    for i in os.listdir(directory+"/"+name):
        data.append(img_to_array(load_img(directory+"/"+name+"/"+str(i))))
        labels.append(name)


le = LabelEncoder()
labels = le.fit_transform(labels)

data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 10)

(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.3, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=trainData.shape[1:]))
model.add(Dense(64, activation='relu', name='dense_one'))
model.add(Dropout(0.5, name='dropout_one'))
model.add(Dense(64, activation='relu', name='dense_two'))
model.add(Dropout(0.5, name='dropout_two'))
model.add(Dense(10, activation='sigmoid', name='output'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(trainData, trainLabels,nb_epoch=50, batch_size=128)
model.save_weights("weights")


print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))




