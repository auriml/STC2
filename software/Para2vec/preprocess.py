from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick
import pandas as pd
import csv

#Generate Biomedical_indext.txt, a file (one sentence for each line) using indexes instead of words
textFile = "/Users/aureliabustos/IdeaProjects/Tesis_tests/report_sentences_preprocessed.csv" #'../../dataset/Biomedical.txt'
df = pd.read_csv(textFile , keep_default_na=False)
#Save file with reports splitted in sentences so that each sentence is a line
sentences =  df['v_preprocessed'].str.split(' \\.')
sentences = [item  for sublist in sentences for item in sublist if item ]
f = open('Biomedical.txt', 'w', newline="")
f.writelines('\n'.join(sentences))

#Save same file (one sentence for each line) using indexes instead of words
# estimate the size of the vocabulary
words = set(text_to_word_sequence(' '.join(sentences)))
vocab_size = len(words)
# integer encode the document
result = [hashing_trick(sentence, round(vocab_size*1.3), hash_function='md5') for sentence in sentences]
f = open('Biomedical_index.txt', 'w', newline="")
writer = csv.writer(f, delimiter= ' ')
writer.writerows(result)