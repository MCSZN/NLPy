import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

# words with imdb dataset
# read positive data
with open('./test/pos_tweets.txt', 'r') as infile:
    pos_tweets = infile.readlines()

# read negative data
with open('./test/neg_tweets.txt', 'r') as infile:
    neg_tweets = infile.readlines()

#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

# separate data into training and test
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2, random_state=42)

data = np.concatenate((pos_tweets, neg_tweets))

# should be improved, as it's very simple
def clean_data(corpus): 
	corpus = [d.lower().split() for d in corpus]
	return corpus


X_train = clean_data(X_train)
X_test = clean_data(X_test)
emb_size = 128

#Initialize model_w2v and build vocabularies
model_w2v = Word2Vec(size=emb_size, min_count=5)
model_w2v.build_vocab(X_train) # should use the whole dataset, not just training set
#Train the model_w2v over train_reviews (this may take several minutes)

model_w2v.train(X_train, total_examples=model_w2v.corpus_count, epochs=2000)
print("training w2v: done")
# save the model_w2v
model_w2v_path = 'test/model_w2v_imdb.bin' 
model_w2v.save(model_w2v_path)

# reload the trained model_w2v
new_model = Word2Vec.load(model_w2v_path)
X = new_model[new_model.wv.vocab] # get words


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
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

X_train = np.concatenate([build_word2vec_from_text(model_w2v, d, emb_size) for d in X_train])

#Train word2vec on test set
#model_w2v.train(x_test, total_examples=model_w2v.corpus_count, epochs=1000)

X_test = np.concatenate([build_word2vec_from_text(model_w2v, d, emb_size) for d in X_test])

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# you can test with MinMaxNormalization
scale = StandardScaler()
scale.fit(X_train)
train_fet = scale.transform(X_train)
test_fet = scale.transform(X_test)

lr = LogisticRegression(penalty='l2', C=0.001)
lr.fit(train_fet, y_train)
print('Test accuracy: %.2f' % lr.score(test_fet, y_test))
print('Train accuracy: %.2f' % lr.score(train_fet, y_train))



