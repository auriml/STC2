import scipy.io as spio

mat = spio.loadmat('../../dataset/Biomedical-STC2.mat', squeeze_me=True)
_mat = spio.loadmat('../../_dataset/Biomedical-STC2.mat', squeeze_me=True)
lite = spio.loadmat('../../dataset/Biomedical-lite.mat', squeeze_me=True)
_lite = spio.loadmat('../../_dataset/Biomedical-lite.mat', squeeze_me=True)


#df = pd.read_csv("../Para2vec/Biomedical_index.txt",  header=None, delimiter= ' ' )
#print(df.head())

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick, Tokenizer

import pandas as pd

df_clusters = pd.read_csv("../Para2vec/sentence_clusters.csv",   delimiter= ',' )
df_clusters.sort_values(['key'],inplace = True)
labels_All = df_clusters['class'] #array of class labels ordered by sentence

textFile = "/Users/aureliabustos/IdeaProjects/Tesis_tests/report_sentences_preprocessed.csv" #'../../dataset/Biomedical.txt'
df = pd.read_csv(textFile , keep_default_na=False)
#Save file with reports splitted in sentences so that each sentence is a line
sentences =  df['v_preprocessed'].str.split(' \\.')
sentences = [item  for sublist in sentences for item in sublist if item ]
#sentences = ['con sin derech', 'no estoy bien', 'pero mañana hara frio', 'yo tengo frio', 'tu como estas']

#Save same file (one sentence for each line) using indexes instead of words
# estimate the size of the vocabulary
words = set(text_to_word_sequence(' '.join(sentences)))
size_vocab =   len(words) + 1 #added one more token to use it for padding sentences to fixed length
# integer encode the document
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)


df_vectors = pd.read_csv("../Para2vec/word_vectors.txt",  header=None, skiprows= 2, delimiter= ' ', na_filter = False )
index = tokenizer.word_index.keys()

d = tokenizer.word_index
#save word dictionary
with open('../../dataset/Biomedical_vocab2idx.dic', 'w+') as file:
        for key, value in d.items():
                file.write("{0}\t{1}\n".format( key , value))

file.close()

df_vectors[49] = df_vectors[0].map(d)
df_vectors = df_vectors[df_vectors[49].notna()]
#df_vectors.sort_values(['index'],inplace = True)
vocab_emb_Word2vec_48_index = df_vectors[49].astype('float64') #array of numbers of words index starting from 1 that corresponds to the wordembedding at each row index  (the size is the number of wordembeddings and should be equal or less than size_vocab)

vocab_emb_Word2vec_48 = df_vectors.transpose()[1:-1] #array of 48 x n_words in dictionary
vocab_emb_Word2vec_48.columns = [i for i in range(vocab_emb_Word2vec_48.shape[1])]
vocab_emb_Word2vec_48  = vocab_emb_Word2vec_48.astype('float64')

biomedical_index = tokenizer.texts_to_sequences(texts = sentences)
all_lbl = [len(i) for i in biomedical_index] #array of number of words on each sentence ordered by sentence
df = pd.DataFrame(biomedical_index)
df.replace([None], [size_vocab], inplace = True )
all = df.apply(pd.to_numeric)
sent_length = all.shape[1]

fea_All = tokenizer.texts_to_matrix(texts = sentences)
from scipy.sparse import csc_matrix
fea_All = csc_matrix(fea_All)

import numpy as np

dict = {'all': all.as_matrix(), 'fea_All' : fea_All,
        'all_lbl':np.asarray([[],all_lbl]), 'labels_All': labels_All.as_matrix(),
        'vocab_emb_Word2vec_48': vocab_emb_Word2vec_48.as_matrix(), 'vocab_emb_Word2vec_48_index' : vocab_emb_Word2vec_48_index.as_matrix(),
        'sent_length':sent_length, 'size_vocab':size_vocab, 'index': [i.strip() for i in list(index)]}

mat3 = spio.savemat("../../dataset/Biomedical-STC2.mat", dict)

import random
idx = np.arange(len(sentences))
random.shuffle(idx)

testIdx = idx[int(len(sentences)*0.8): ]#array of idxs of sentences -20% sample
trainIdx = idx[:int(len(sentences)*0.8)]#array of idxs of sentences -80% sample
fea = fea_All #csc matrix
gnd = labels_All.as_matrix() #array of sentence classes
dict = {'testIdx' : testIdx, 'trainIdx': trainIdx,
        'fea': fea_All, 'gnd': gnd}
mat4 = spio.savemat("../../dataset/Biomedical-lite.mat", dict)

