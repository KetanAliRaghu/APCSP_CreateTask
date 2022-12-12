import preprocess_kgptalkie as kgp
import pandas as pd

df = pd.read_csv('data/emotion-emotion_69k.csv')

for i in range(len(df['empathetic_dialogues'])):
    df['empathetic_dialogues'][i] = df['empathetic_dialogues'][i].replace('Customer :' , '').replace('Agent :' , '').replace(',' , '')
    df['empathetic_dialogues'][i] = kgp.remove_accented_chars(df['empathetic_dialogues'][i])

df.to_csv('data/emotion_69k_preprocess.csv')
