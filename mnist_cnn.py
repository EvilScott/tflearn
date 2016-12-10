import tflearn as tfl
from tflearn.datasets import mnist

# grab and reshape data
data, labels, test_data, test_labels = mnist.load_data(one_hot=True)
data = data.reshape([-1, 28, 28, 1])
test_data = test_data.reshape([-1, 28, 28, 1])

# set up network
network = tfl.input_data(shape=[None, 28, 28, 1])
network = tfl.conv_2d(network, 32, 3, activation='relu', regularizer='L2')
network = tfl.max_pool_2d(network, 2)
network = tfl.local_response_normalization(network)
network = tfl.conv_2d(network, 64, 3, activation='relu', regularizer='L2')
network = tfl.max_pool_2d(network, 2)
network = tfl.local_response_normalization(network)
network = tfl.fully_connected(network, 128, activation='tanh')
network = tfl.dropout(network, 0.8)
network = tfl.fully_connected(network, 256, activation='tanh')
network = tfl.dropout(network, 0.8)
network = tfl.fully_connected(network, 10, activation='softmax')
network = tfl.regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

# create model
model = tfl.DNN(network)

# train model
model.fit(data,
          labels,
          n_epoch=5,
          validation_set=(test_data, test_labels),
          show_metric=True,
          batch_size=500)
