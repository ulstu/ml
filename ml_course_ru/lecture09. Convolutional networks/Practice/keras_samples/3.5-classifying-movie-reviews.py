
# coding: utf-8

import keras
keras.__version__


# # Classifying movie reviews: a binary classification example
# 
# We'll be working with "IMDB dataset", a set of 50,000 highly-polarized reviews from the Internet Movie Database. They are split into 25,000
# reviews for training and 25,000 reviews for testing, each set consisting in 50% negative and 50% positive reviews.
# 
# Why do we have these two separate training and test sets? You should never test a machine learning model on the same data that you used to 
# train it! Just because a model performs well on its training data doesn't mean that it will perform well on data it has never seen, and 
# what you actually care about is your model's performance on new data (since you already know the labels of your training data -- obviously 
# you don't need your model to predict those). For instance, it is possible that your model could end up merely _memorizing_ a mapping between 
# your training samples and their targets -- which would be completely useless for the task of predicting targets for data never seen before. 
# We will go over this point in much more detail in the next chapter.
# 
# Just like the MNIST dataset, the IMDB dataset comes packaged with Keras. It has already been preprocessed: the reviews (sequences of words) 
# have been turned into sequences of integers, where each integer stands for a specific word in a dictionary.
# 
# The following code will load the dataset (when you run it for the first time, about 80MB of data will be downloaded to your machine):


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# 
# The argument `num_words=10000` means that we will only keep the top 10,000 most frequently occurring words in the training data. Rare words 
# will be discarded. This allows us to work with vector data of manageable size.
# 
# The variables `train_data` and `test_data` are lists of reviews, each review being a list of word indices (encoding a sequence of words). 
# `train_labels` and `test_labels` are lists of 0s and 1s, where 0 stands for "negative" and 1 stands for "positive":

#train_data[0]
train_labels[0]


# Since we restricted ourselves to the top 10,000 most frequent words, no word index will exceed 10,000:
max([max(sequence) for sequence in train_data])


# For kicks, here's how you can quickly decode one of these reviews back to English words:
# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)


# ## Preparing the data
# 
# 
# We cannot feed lists of integers into a neural network. We have to turn our lists into tensors. There are two ways we could do that:
# 
# * We could pad our lists so that they all have the same length, and turn them into an integer tensor of shape `(samples, word_indices)`, 
# then use as first layer in our network a layer capable of handling such integer tensors (the `Embedding` layer, which we will cover in 
# detail later in the book).
# * We could one-hot-encode our lists to turn them into vectors of 0s and 1s. Concretely, this would mean for instance turning the sequence 
# `[3, 5]` into a 10,000-dimensional vector that would be all-zeros except for indices 3 and 5, which would be ones. Then we could use as 
# first layer in our network a `Dense` layer, capable of handling floating point vector data.
# 
# We will go with the latter solution. Let's vectorize our data, which we will do manually for maximum clarity:
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)


# Here's what our samples look like now:
x_train[0]


# We should also vectorize our labels, which is straightforward:
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


history_dict = history.history
history_dict.keys()


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)



results

model.predict(x_test)

