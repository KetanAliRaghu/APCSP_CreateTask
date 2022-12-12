import pandas as pd
from datasets import load_dataset

"""
df = pd.read_csv('data/tweet_emotions.csv')

for i in range(len(df['content'])):
    tokens = df['content'][i].split(' ')
    for j in reversed(range(len(tokens))):
        if tokens[j].startswith('@'):
            tokens[j] = '@user'
    df['content'][i] = ' '.join(tokens)

print(df.head())
df.to_csv('data/tweet_emotions_genmentions.csv')
"""

"""
df = pd.read_csv('data/tweet_emotions_genmentions.csv')
df = df[df.sentiment != 'empty']

print(df['sentiment'].value_counts())

df_neutral = df[df['sentiment'] == 'neutral'].sample(8000 , replace = True)
df_anger = df[df['sentiment'] == 'anger'].sample(7500 , replace = True)
df_happiness = df[df['sentiment'] == 'happiness'].sample(7500 , replace = True)
df_sadness = df[df['sentiment'] == 'sadness'].sample(7500 , replace = True)
df_relief = df[df['sentiment'] == 'relief'].sample(7000 , replace = True)
df_worry = df[df['sentiment'] == 'worry'].sample(7000 , replace = True)
df_love = df[df['sentiment'] == 'love'].sample(6000 , replace = True)
df_surprise = df[df['sentiment'] == 'surprise'].sample(6000 , replace = True)
df_fun = df[df['sentiment'] == 'fun'].sample(6000 , replace = True)
df_hate = df[df['sentiment'] == 'hate'].sample(6000 , replace = True)
df_enthusiasm = df[df['sentiment'] == 'enthusiasm'].sample(6000 , replace = True)
df_boredom = df[df['sentiment'] == 'boredom'].sample(6000 , replace = True)

df_new = pd.concat([
    df_neutral,
    df_anger,
    df_happiness,
    df_sadness,
    df_relief,
    df_worry,
    df_love,
    df_surprise,
    df_fun,
    df_hate,
    df_enthusiasm,
    df_boredom
]).sample(frac = 1).reset_index(drop = True).drop(['tweet_id'] , axis = 1)

index_dict = {
    'neutral': 0,
    'anger': 1,
    'happiness': 2,
    'sadness': 3,
    'relief': 4,
    'worry': 5,
    'love': 6,
    'surprise': 7,
    'fun': 8,
    'hate': 9,
    'enthusiasm': 10,
    'boredom': 11
}

df_new['sent_vec'] = df_new['sentiment'].replace(index_dict , inplace = False)

df_new.to_csv('data/tweet_emotions_genmentions_sampled.csv')

print(df.head())
print(df_new.head())
"""

df_train = pd.DataFrame(load_dataset('emotion' , split = 'train'))
df_test = pd.DataFrame(load_dataset('emotion' , split = 'test'))
df_val = pd.DataFrame(load_dataset('emotion' , split = 'validation'))

print(df_train.head())
print(df_train['label'].value_counts())
print(df_train.shape)

print(df_test.head())
print(df_test['label'].value_counts())
print(df_test.shape)

print(df_val.head())
print(df_val['label'].value_counts())
print(df_val.shape)

df = pd.concat([df_train , df_test , df_val]).reset_index(drop = True)
df = df[df['label'] != 2]
df = df[df['label'] != 5]

df['label'].replace(4 , 2 , inplace = True)

print(df.head())
print(df['label'].value_counts())

df_sad = df[df['label'] == 0].sample(6000 , replace = True)
df_joy = df[df['label'] == 1].sample(6000 , replace = False)
df_fear = df[df['label'] == 2].sample(6000 , replace = True)
df_anger = df[df['label'] == 3].sample(5000 , replace = True)

df_full = pd.concat([df_sad , df_joy , df_fear , df_anger]).sample(frac = 1).reset_index(drop = True)
df_full.to_csv('data/emotion_4class_preprocessed.csv')

print(df_full['label'].value_counts())
print('done')
