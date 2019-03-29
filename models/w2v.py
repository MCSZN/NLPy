import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
data= pd.read_csv('../inputs/spam.csv', sep=",", encoding = "ISO-8859-1").drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)


def clean_data(corpus): 
    corpus = [d.lower().split() for d in corpus]
    return corpus


def build_word2vec_from_text(model_w2v, sentence, emb_size):
    emb_vec = np.zeros(emb_size).reshape((1, emb_size))
    count = 0.
    for word in sentence:
        try:
            emb_vec += model_w2v[word].reshape((1, emb_size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        emb_vec /= count
    return emb_vec


X_train, X_test, y_train, y_test = train_test_split(data.v2.values, data.v1.values, test_size=0.2, random_state=42)
X_train = clean_data(X_train)
X_test = clean_data(X_test)
emb_size = 128

model_w2v = Word2Vec(size=emb_size, min_count=5)
model_w2v.build_vocab(X_train)
model_w2v.train(X_train, total_examples=model_w2v.corpus_count, epochs=20)


X = new_model[model_w2v.wv.vocab] # get words
X_train = np.concatenate([build_word2vec_from_text(model_w2v, d, emb_size) for d in X_train])
X_test = np.concatenate([build_word2vec_from_text(model_w2v, d, emb_size) for d in X_test])

model = GaussianNB()
pred = model.fit(X_train, y_train)

pred.score(X_test, y_test)





