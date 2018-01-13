import pandas as pd


df = pd.read_csv('Biomedical_para2vecs.txt', names = ['key', 'class'], sep = ' ' )
print(df.head())
textFile = "Biomedical.txt"
df1 = pd.read_csv(textFile, names = ['text'])
df1['key'] = df1.index
print(df1.head())
df['key'] = df['key'].str.replace('_\\*',  '')
df['key'] = pd.to_numeric(df['key'])
print(df.head())
dfjoin = pd.merge(df,df1, on = 'key', how='outer')
dfjoin = dfjoin.sort_values(by=['class'])

dfjoin.to_csv("sentence_clusters.csv", columns=['key','class', 'text'], encoding="ISO-8859-1")





