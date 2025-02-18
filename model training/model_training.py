import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import pickle
import json

import nltk
import numpy as np
import tensorflow

from nltk.stem import WordNetLemmatizer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 


lemmatizer = WordNetLemmatizer()

intents = json.load(open('model_training/intents.json'))

words = []
classes = []
documents = []
ignore_symbols = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]

words = sorted(set(words))
classes = sorted(set(classes))


if not os.path.exists('model'):
    os.makedirs('model')

pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns =  [lemmatizer.lemmatize(word) for word in word_patterns if word not in ignore_symbols]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
# Split training into X and y before converting to numpy array
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Convert to numpy arrays after splitting
train_x = np.array(train_x)
train_y = np.array(train_y)


model = Sequential()

model.add(Dense(128, input_shape=(len(words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

sgd = SGD(learning_rate = 0.01, weight_decay = 1e-9, momentum = 0.9, nesterov = True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs= 200, batch_size=5, verbose=1)

model.save('model/chatbot_model.keras')