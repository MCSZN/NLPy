from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np

max_features = 500

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(32))  
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# complile
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, max_features))
y_train = np.random.randint(2, size=(1000, 1))

# Generate dummy validation data
x_val = np.random.random((100, max_features))
y_val = np.random.randint(2, size=(100, 1))

model.fit(x_train, y_train, batch_size=24, epochs=1)
score = model.evaluate(x_val, y_val, batch_size=24)

print(score)