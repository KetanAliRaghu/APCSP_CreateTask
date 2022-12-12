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
fifo = FIFO(5)
fifo.append('Hello')
fifo.append(text)

idx = 0
AI_last_sent = FIFO(2 , ['' , ''])

while text != 'done':
    if idx == 5:
        fifo.change_max(4)
    elif idx == 2:
        fifo.change_max(3)

    # print(list(fifo))
    # print(' '.join(fifo))
    inputs = tokenizer(' '.join(fifo) , return_tensors = 'pt' , padding = True)
    reply_ids = model.generate(**inputs)

    AI_output = tokenizer.batch_decode(reply_ids)[-1]
    AI_output = kgp.remove_html_tags(AI_output)
    AI_output = sent_tokenize(AI_output)

    # print(AI_output)

    if idx > 8 and fifo.get_max() > 3:
        if len(AI_output) >= 2 and len(AI_output[-1]) < 6 and len(AI_output[-2]) < 6 and AI_output[-2] + AI_output[-1] != AI_last_sent[0]:
            AI_output = AI_output[-2] + AI_output[-1]
        elif len(AI_output) == 1 or AI_output[-1] != AI_last_sent[1] or AI_output[-1] != AI_last_sent[0]:
            AI_output = AI_output[-1]
        else:
            AI_output = AI_output[-2]
    else:
        AI_output = ' '.join(AI_output)


    print(f'BOT: {AI_output}')
    text = input('USER: ')

    fifo.append(text)
    AI_last_sent.append(AI_output)
    idx += 1
