import pandas as pd
from keras.saving.save import load_model

df = pd.read_csv('data/emotion_69k_preprocess4.csv')
model = load_model('models/EmDetModel_4Class_20Epoch_88Acc')


