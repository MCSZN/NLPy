#Counter Vectorization

from sklearn.feature_extraction.text import CountVectorizer

texts = ["Ramiess sings classic songs", "he listens to old pop ", "and rock music", ' and also listens to classical songs']

cv = CountVectorizer()

# tokenize and build vocab
cv.fit(texts)
# summarize
print(cv.vocabulary_)
# encode document
vector = cv.transform(texts)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

# transform new text
text2 = ["the puppy"]
vector = cv.transform(text2)
print(vector.toarray())

# both in the same time
cv1 = CountVectorizer()
cv_fit = cv1.fit_transform(texts)
print(cv1.get_feature_names())
print(len(cv1.get_feature_names()))
print(cv_fit.toarray())




