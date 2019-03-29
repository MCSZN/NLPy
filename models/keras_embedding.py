import keras
import collections
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout
from sklearn.model_selection import train_test_split

data= pd.read_csv('../inputs/spam.csv', sep=",", encoding = "ISO-8859-1").drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

vocab = list({word for sentence in data.v2 for word in sentence})
stoi = collections.defaultdict(lambda: len(vocab),{string:integer for integer,string in enumerate(vocab)})
data['v2'] = data.v2.apply(lambda l: np.array([stoi[s] for s in l]))
data.replace(['ham', 'spam'], [0,1], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data.v2.values, data.v1.values, test_size=0.2, random_state=42)
X_train= pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

max_features = 500

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))  
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train)
model.evaluate(X_test, y_test)