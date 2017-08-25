import cv2
import time
import datetime
import numpy as np

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image as im_preproc


def classify(sample):
    x = []
    # height, width = sample.shape[0: 2]
    # print height, width
    sample = cv2.resize(sample, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('ROI', sample)
    cv2.waitKey(0)

    sample = im_preproc.img_to_array(sample)
    # this is example of general case
    x.append(sample)

    x = np.array(x)
    # because OpenCV 224, 224, 3
    x = x.reshape((1, 3, 224, 224))
    x = x.astype(np.float32)
    # print x.shape

    x = preprocess_input(x)
    # print x.shape

    preds = model.predict(x, verbose=1)

    feat = decode_predictions(preds, top=5)[0]
    # print feat
    text = ''
    for f in feat:
        if f[2] > min_probability:
            text += ' Class: ' + f[1] + ', p=' + str(f[2]) + ' || '

    return text


model = VGG19(weights='imagenet', include_top=True)
min_probability = 0.1
image = cv2.imread('car_1.jpg')
text = classify(image)
print text

image = cv2.imread('car_2.jpg')
text = classify(image)
print text

image = cv2.imread('car_3.jpg')
text = classify(image)
print text

image = cv2.imread('cat_1.png')
text = classify(image)
print text

image = cv2.imread('elephant.jpg')
text = classify(image)
print text

image = cv2.imread('plate.jpg')
text = classify(image)
print text

image = cv2.imread('apple.jpg')
text = classify(image)
print text
cv2.destroyAllWindows()
