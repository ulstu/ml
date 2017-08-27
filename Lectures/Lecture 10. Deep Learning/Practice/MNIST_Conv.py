import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import model_from_json


def SaveToJSon(model, fileName):
    model_json = model.to_json()
    with open(fileName + ".net", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(fileName + ".wgt")
    print("model saved to disk")

def LoadFromJSon(fileName):
    json_file = open(fileName + ".net", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(fileName + ".wgt")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    print("Loaded model from disk")

    return loaded_model

def CreateModel():
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(2 * nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

np.random.seed(1337)
(X_train, y_train), (X_test, y_test) = mnist.load_data()


batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

for ind in range(15):
    plt.subplot(3,5,ind + 1)
    plt.imshow(X_train[ind], cmap=plt.get_cmap('gray'))

plt.show()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = CreateModel()

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss trained Model:', score[0])
print('Test accuracy trained Model:', score[1])


SaveToJSon(model, "cnet")
model = LoadFromJSon("cnet")
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss loaded Model:', score[0])
print('Test accuracy loaded Model:', score[1])

model_2 = CreateModel()
score = model_2.evaluate(X_test, Y_test, verbose=0)
print('Test loss new Model:', score[0])
print('Test accuracy  new Model:', score[1])