"""
Edureka
"""

import os
import math
import json

import tensorflow as tf
import  numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from DataStructures import FIFO

import keras
from keras.preprocessing.text import Tokenizer
from keras import Model
from keras.layers import Input , Embedding , LSTM,\
    Dense , GlobalMaxPooling1D , Flatten , SimpleRNN , Bidirectional
import matplotlib.pyplot as plt
from keras.utils import pad_sequences
from keras import losses , optimizers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import preprocess_kgptalkie as kgp

df = pd.read_csv('data/emotion_69k_preprocess.csv')

tokenizer = Tokenizer(num_words = 18000)
tokenizer.fit_on_texts(df['empathetic_dialogues'])

train = tokenizer.texts_to_sequences(df['empathetic_dialogues'])
X = pad_sequences(train)

labEnc = LabelEncoder()
y = labEnc.fit_transform(df['labels'])

input_shape = X.shape[1]
vocab = len(tokenizer.word_index)
output_len = labEnc.classes_.shape[0]

print(output_len)

"""
# Model
input_layer = Input(shape = (input_shape,))
x = Embedding(vocab + 1 , 10)(input_layer)
x = LSTM(10 , return_sequences = True)(x)
x = Flatten()(x)
x = Dense(output_len , activation = 'sigmoid')(x)

model = Model(inputs = [input_layer] , outputs = x)
model.compile(
    loss = losses.SparseCategoricalCrossentropy(),
    optimizer = optimizers.Adam(),
    metrics = ['accuracy']
)

print(model.summary())

report = model.fit(X , y , epochs = 12)

model.save('models/GenerationPipelineE_20Epoch')
"""

"""
plt.plot(report.history['accuracy'] , label = 'Accuracy')
plt.plot(report.history['loss'] , label = 'Loss')
plt.legend()
plt.show()
"""

model = keras.saving.save.load_model('models/GenerationPipelineE_20Epoch')

def pad_seq(texts , length):
    returned = []

    if len(texts) <= length:
        for k in range(length - len(text)):
            returned.append(0)

        for l in range(len(texts)):
            returned.append(texts[l])

        return np.array(returned)

    for k in reversed(range(int(len(texts) / length))):
        returned.append(texts[len(texts) - (length * k) - length : len(texts) - (length * k)])
    if len(texts) % length != 0:
        padded = texts[:len(texts) - (length * int(len(texts) / length))]
        for j in range(length - len(padded)):
            padded.insert(0 , 0)
        returned.insert(padded , 0)
    return np.array(returned)

text = 'hello'
text = tokenizer.texts_to_sequences([text])
text = np.array(text).reshape(-1)
text = pad_sequences([text] , 111)
text = model.predict(text)
text = text.argmax()
print(f'BOT: {labEnc.inverse_transform([text])}')

text = input('USER: ')

while text != 'done':
    """
    if idx == 5:
        fifo.change_max(4)
    elif idx == 3:
        fifo.change_max(3)
    """

    # text = fifo.str_concat('. ')
    text = tokenizer.texts_to_sequences([text])
    text = np.array(text).reshape(-1)
    text = pad_sequences([text], 111)
    text = model.predict(text)
    text = text.argmax()
    text = labEnc.inverse_transform([text])[0]

    print(f'BOT: {text}')
    text = input('USER: ')
    # fifo.append(text)
