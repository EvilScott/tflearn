import tflearn as tfl
from tflearn.datasets import mnist

# grab data
data, labels, test_data, test_labels = mnist.load_data(one_hot=True)

# set up network
input_layer = tfl.input_data(shape=[None, 784])
hidden_layer = tfl.fully_connected(input_layer, 300, activation='sigmoid')
dropout = tfl.dropout(hidden_layer, 0.8)
output_layer = tfl.fully_connected(dropout, 10, activation='sigmoid')
regression = tfl.regression(output_layer, learning_rate=0.002)

# create model
model = tfl.DNN(regression)

# train model
model.fit(data,
          labels,
          n_epoch=3,
          validation_set=(test_data, test_labels),
          show_metric=True)
