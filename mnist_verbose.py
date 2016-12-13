import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data
mnist = input_data.read_data_sets('mnist/', one_hot=True)

# weights and biases
W1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[100]))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

# placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# layers
y = tf.nn.relu(tf.matmul(x, W1) + b1)
y = tf.nn.dropout(y, keep_prob)
y = tf.matmul(y, W2) + b2

# model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
while mnist.train.epochs_completed < 5:
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8})

# testing
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
