### Author: Riya Nakarmi ###
### College Project ###

import random  # Python standard library module that provides functions for generating random numbers
import json # Works with JSON data
import pickle # Converts a Python object hierarchy to a byte stream
import numpy as np # A popular numerical computing library that enables users to work with large, multi-dimensional arrays and matrices in Python.

import nltk # Natural Language Processing Library
from nltk.stem import WordNetLemmatizer # Process of reducing inflected forms to base or root form

from tensorflow.keras.models import Sequential # Imports Keras deep learning framework from TensorFlow.
#A model type that allows to build deep learning models layer by layer.
from tensorflow.keras.layers import Dense, Activation, Dropout 
#Dense: Most commonly used neural network layer type, it's essentially a fully connected layer where 
# every neuron is connected to every other neuron in the previous layer.
# Actiation: A layer type taht specifies an activation function for the output of the previous layer.
from tensorflow.keras.optimizers import SGD # Stochastic gradient descent, an optimization algorithm used for training deep learning models

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!',',','.']

# Prepares the input data for use in training an NLP model.
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# Serializes words into binary format using wb mode
pickle.dump(words, open('words.pkl', 'wb'))
# Serializes classes into binary format using wb mode
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag =[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('Done')