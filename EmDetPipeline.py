import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import keras
from keras.layers import Dense , Dropout , Input
from keras.models import Model
from keras import optimizers , losses , callbacks

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# BERT_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
# BERT_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
better_encoder = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4')

df = pd.read_csv('data/emotion_4class_preprocessed.csv')

X_train , X_test , y_train , y_test = train_test_split(df['text'],
                                                       df['label'],
                                                       test_size = 0.25,
                                                       random_state = 2)

class SentimentModel(keras.Model):
    def __init__(self , units , num_outputs):
        super(SentimentModel , self).__init__()
        # self.BERT_preprocess = BERT_preprocess
        # self.BERT_encoder = BERT_encoder
        self.dense1 = Dense(units , activation = 'relu')
        self.dense2 = Dense(units * 2 , activation = 'relu')
        self.dense3 = Dense(units , activation = 'relu')
        self.dropout = Dropout(0.2)
        self.dense4 = Dense(units , activation = 'relu')
        self.dense5 = Dense(units / 2 , activation = 'relu')
        self.out = Dense(num_outputs , activation = 'softmax')

    def call(self, inputs, training = False, mask = None):
        # x = self.BERT_preprocess(inputs)
        # x = self.BERT_encoder(x)
        x = self.dense1(inputs , training = training)
        x = self.dense2(x , training = training)
        x = self.dense3(x , training = training)
        x = self.dropout(x)
        x = self.dense4(x , training = training)
        x = self.dense5(x , training = training)
        return self.out(x)

    def model(self):
        x = Input(shape = () , dtype = tf.string , name = 'input layer')
        return Model(inputs = [x] , outputs = self.call(x))

# Custom Callback Function (lr = learn rate)
def scheduler(epoch , lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.95

lr_scheduler = callbacks.LearningRateScheduler(scheduler , verbose = 1)

print(X_test)
print(y_test)
"""
inputs = Input(shape = () , dtype = tf.string , name = 'input')
x = better_encoder(inputs)
x = Dense(128 , activation = 'relu')(x)
x = Dense(256 , activation = 'relu')(x)
x = Dense(128 , activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(128 , activation = 'relu')(x)
x = Dense(64 , activation = 'relu')(x)
output = Dense(4 , activation = 'softmax')(x)

model = Model(inputs = [inputs] , outputs = output)
model.compile(
    optimizer = optimizers.Adam(),
    loss = losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = 'accuracy'
)
model.fit(X_train , y_train , epochs = 20,
          callbacks = [lr_scheduler])

print(model.evaluate(X_test , y_test))
print(model.predict(['i am happy']))

model.save('models/EmDetModel_4Class_20Epoch_88Acc')
"""

model = keras.saving.save.load_model('models/EmDetModel_4Class_20Epoch_88Acc')
y_pred = model.predict([X_test])

print(y_pred)
# if the value < 0.5 it is set to 0 else it is 1
print(len(y_pred))
y_pred_new = []
for i in range(len(y_pred)):
    max_index = 0
    maximum = 0
    for j in range(len(y_pred[i])):
        if y_pred[i][j] > maximum:
            maximum = y_pred[i][j]
            max_index = j
    y_pred_new.append(max_index)

print(y_pred)

print(classification_report(y_test , y_pred_new))
