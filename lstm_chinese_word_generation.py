'''
    Script to generate text from Chinese Novels

    With a word level corpus language model

'''

from __future__ import print_function
import numpy as np
import random, sys
import codecs

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import load_model

import re
import jieba
import collections
import tensorflow as tf

def readfile(file_path):
    f = codecs.open(file_path, 'r', 'utf-8')
    alltext = f.read()
    alltext = re.sub(r'\s', '', alltext)
    seglist = list(jieba.cut(alltext, cut_all=False))
    return seglist


def build_vocab(filename):
    data = readfile(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    #print('word',words)
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    dataids = []
    for w in data:
        dataids.append(word_to_id[w])
    return word_to_id, id_to_word, dataids


word_to_id,id_to_word,dataids = build_vocab('book1.txt')

vocab = set(dataids)

# build training X and Y, using one-hot vectorization
maxlen = 20
step = 30
sentences = []
next_words = []
for i in range(0, len(dataids) - maxlen, step):
    sentences.append(dataids[i : i + maxlen])
    next_words.append(dataids[i + maxlen])
#print('sentence size',len(sentences))
#print('sentences',sentences)
#print('next_words',next_words)

X = np.zeros((len(sentences), maxlen, len(vocab)),dtype=np.bool)

Y = np.zeros((len(sentences), len(vocab)),dtype=np.bool)

#print(X.shape)
#print(Y.shape)

for i, sentence in enumerate(sentences):
    #print(i,sentence)
    for t, word in enumerate(sentence):
        X[i, t, word] = 1.
    Y[i, next_words[i]] = 1.

# build or load model
model_path = 'lstm_chinese_word_model.h5'

def buildModel():
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(vocab))))
    model.add(Dropout(0.2))

    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(len(vocab)))
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

    start_index = random.randint(0, len(dataids) - maxlen - 1)

    for diversity in [0.2, 0.4, 0.6, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        tempStr = ''
        sentence = dataids[start_index: start_index + maxlen]
        for index in range(0,maxlen):
            word = id_to_word[sentence[index]]
            #print(word)
            tempStr += word
        #print('tempStr',tempStr)
        generated += tempStr
        print('----- Generating with seed: "' + tempStr + '"')
        sys.stdout.write(generated)

        for iteration in range(400):
            x = np.zeros((1, maxlen, len(vocab)))
            for t, word in enumerate(sentence):
                x[0, t, word] = 1.

            preds = model.predict(x, verbose=0)[0]

            next_index = sample(preds, diversity)
            next_word = id_to_word[next_index]

            generated += next_word

            sentence = sentence[1:]

            sentence.append(next_index)

            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
