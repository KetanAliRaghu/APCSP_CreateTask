import pandas as pd

df = pd.read_json('data/MoviePreferencesConversationalData.json')
print(df.head())

final_df = pd.DataFrame({
    'CONVERSATION_ID': [],
    'USER': [],
    'ASSISTANT': []
})

for conv in df['utterances']:
    new_df = {
        'CONVERSATION_ID': [],
        'USER': [],
        'ASSISTANT': []
    }

    for i in conv:
        last = ''
        if i['speaker'] == last:
            new_df[i['speaker']][len(new_df[i['speaker']]) - 1] += '. ' + i['text']
        else:
            new_df[i['speaker']].append(i['text'])
            new_df['CONVERSATION_ID'] = i['index']

    if len(new_df['USER']) > len(new_df['ASSISTANT']):
        new_df['ASSISTANT'].append('')
    elif len(new_df['USER']) < len(new_df['ASSISTANT']):
        new_df['USER'].append('')

    final_df.join(pd.DataFrame(new_df)).reset_index(drop = True)

print(final_df)
