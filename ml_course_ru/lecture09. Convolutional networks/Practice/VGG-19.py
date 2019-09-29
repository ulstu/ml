import cv2
import numpy as np
import os

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions


def classify(filename):
    print('\n---------------------------------')
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    feat = decode_predictions(preds, top=5)[0]
    text = ''
    for f in feat:
        if f[2] > min_probability:
            text += 'filename: {}; class: {}; probability: {}\n'.format(filename, f[1], str(f[2]))

    return text

model = VGG19(weights='imagenet', include_top=True)
min_probability = 0.1

files = [x for x in os.listdir('./') if x.endswith('.jpg') or x.endswith('.png')]
for file in files:
    print (classify(file))

cv2.destroyAllWindows()
