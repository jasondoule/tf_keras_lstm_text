'''
    Script to generate peom text from Chinese Peoms

    With a char level corpus language model

'''

from __future__ import print_function
import numpy as np
import random, sys
import codecs

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model


#read file get corpus and vacab
f = codecs.open('poem.txt', 'r', 'utf-8')
text = f.read()
print('corpus length:', len(text))
chars = set(text)
print('total chars:', len(chars))

#build index dict
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build training X and Y, using one-hot vectorization
maxlen = 20
step = 15
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("size:",len(sentences)*maxlen*len(chars))
X = np.zeros((len(sentences), maxlen, len(chars)),dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)),dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1.
    Y[i, char_indices[next_chars[i]]] = 1.

# build or load model
model_path = 'lstm_chinese_poem_model.h5'

def buildModel():
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))

    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    return model

try:
    model = load_model(model_path)
except:
    model = buildModel()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i


# train the model, output generated text
for iteration in range(1, 50):
    print()
    print('-' * 80)
    print('Iteration', iteration)
    model.fit(X, Y, batch_size=128, epochs=1)
    print('Save model...')
    model.save(model_path)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.4, 0.6, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]

            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
