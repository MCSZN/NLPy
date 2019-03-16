
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
texts = ["Ramiess sings classic songs", "he listens to old pop ", "and rock music", 
		' and also listens to classical songs']
# create the transform
vc_tf_idf = TfidfVectorizer()
# tokenize and build vocab
vc_tf_idf.fit(texts)
# summarize
print(vc_tf_idf.vocabulary_)
print(vc_tf_idf.idf_)
# encode document
vector = vc_tf_idf.transform([texts[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

