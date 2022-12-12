"""
Transfer Learning
Conversational Model Therapy (CMT)
"""

import tensorflow as tf
import tensorflow_hub as hub

from DataStructures import FIFO
from nltk.tokenize import sent_tokenize
import preprocess_kgptalkie as kgp

import keras
from keras.layers import Dense , Dropout , Layer,\
    LSTM , SimpleRNN , Input
from keras import optimizers , losses , callbacks
from keras import Model ,  Sequential

import transformers
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM,\
    BlenderbotTokenizer , TFBlenderbotModel , TFBlenderbotForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

inputs = tokenizer(['Hello!'] , return_tensors = 'pt' , padding = True)
reply_ids = model.generate(**inputs)
print(tokenizer.batch_decode(reply_ids))


def remove_instances(string: str , regex: str):
    while regex in string:
        string = string[0 : string.index(regex):] + string[len(regex):]
    return string

text = input('USER: ')

while text != 'done':
    # print(list(fifo))
    # print(' '.join(fifo))
    inputs = tokenizer(text , return_tensors = 'pt' , padding = True)
    reply_ids = model.generate(**inputs)

    AI_output = tokenizer.batch_decode(reply_ids)[-1]
    AI_output = kgp.remove_html_tags(AI_output)
    AI_output = sent_tokenize(AI_output)

    print(f'BOT: {AI_output}')
    text = input('USER: ')
