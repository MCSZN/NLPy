
import nltk
from nltk import word_tokenize
sentence= "This course is about Deep Learning and Natural Language Processing!"
tokens = word_tokenize(sentence)
print("before: ", tokens)

# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
new_tokens = [w for w in tokens if not w in stop_words]

print("after removing stop words: ", new_tokens)

