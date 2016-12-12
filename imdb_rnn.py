import tflearn as tfl
from tflearn.datasets import imdb
from tflearn.data_utils import to_categorical, pad_sequences

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tfl.input_data([None, 100])
net = tfl.embedding(net, input_dim=10000, output_dim=128)
net = tfl.lstm(net, 128, dropout=0.8)
net = tfl.fully_connected(net, 2, activation='softmax')
net = tfl.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

# Training
model = tfl.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)
