from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import RegexpTokenizer  
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


x_train = ["Ramiess sings classic songs", "he listens to old pop ", "and rock music",
         ' and also listens to classical songs']

def process(input_text):
    # Create a regular expression tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Create a Snowball stemmer 
    stemmer = SnowballStemmer('english')

    # Get the list of stop words 
    stop_words = stopwords.words('english')
    
    # Tokenize the input string
    tokens = tokenizer.tokenize(input_text.lower())

    # Remove the stop words 
    tokens = [x for x in tokens if not x in stop_words]
    
    # Perform stemming on the tokenized words 
    tokens_stemmed = [stemmer.stem(x) for x in tokens]

    return tokens_stemmed


print(x_train)

cv = CountVectorizer()
tfidf = TfidfTransformer(smooth_idf=False)

cv.fit(x_train)
x_train_df = cv.transform(x_train)
print(x_train_df)

x_train_df = tfidf.fit_transform(x_train_df)
print(x_train_df)

