import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

print('Getting MNIST Dataset...')
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('Data Extracted.')

# Hyperparameters - Tune as needed
learning_rate = 0.001
training_epochs = 20
batch_size = 128

display_step = 1

# Declaring the size of layers
layer_1_weight_shape = (784, 256)
layer_2_weight_shape = (256, 128)
layer_3_weight_shape = (128, 10)

# Initialisation of Zero Weights
zero_weights_1 = tf.Variable(tf.zeros(layer_1_weight_shape))
zero_weights_2 = tf.Variable(tf.zeros(layer_2_weight_shape))
zero_weights_3 = tf.Variable(tf.zeros(layer_3_weight_shape))

# Initialisation of One Weights
one_weights_1 = tf.Variable(tf.ones(layer_1_weight_shape))
one_weights_2 = tf.Variable(tf.ones(layer_2_weight_shape))
one_weights_3 = tf.Variable(tf.ones(layer_3_weight_shape))

# Best Practice Weights are random with range +/- 1/sqrt(n) where n is number of input to neuron
# For this example just take the value of -1, 1
bp_weights_1 = tf.Variable(tf.random_uniform(layer_1_weight_shape, -1, 1))
bp_weights_2 = tf.Variable(tf.random_uniform(layer_2_weight_shape, -1, 1))
bp_weights_3 = tf.Variable(tf.random_uniform(layer_3_weight_shape, -1, 1))

# Biases
# Initialisation of Zero biases
bias_weights_1 = tf.Variable(tf.zeros(layer_1_weight_shape[1]))
bias_weights_2 = tf.Variable(tf.zeros(layer_2_weight_shape[1]))
bias_weights_3 = tf.Variable(tf.zeros(layer_3_weight_shape[1]))

# Placeholders (Graph Inputs)
x = tf.placeholder('float', [None, 28, 28, 1])
y = tf.placeholder('float', [None, 10])

# Reshaped x
x_flat = tf.reshape(x, [-1, 784])

# Different networks for different layers

# 1 Zero weights
zero_hidden_layer_1 = tf.add(tf.matmul(x_flat, zero_weights_1), bias_weights_1)
zero_hidden_layer_1 = tf.nn.relu(zero_hidden_layer_1)

zero_hidden_layer_2 = tf.add(tf.matmul(zero_hidden_layer_1, zero_weights_2), bias_weights_2)
zero_hidden_layer_2 = tf.nn.relu(zero_hidden_layer_2)

zero_logits = tf.add(tf.matmul(zero_hidden_layer_2, zero_weights_3), bias_weights_3)

# Zero Cost
zero_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= zero_logits, labels= y))

# Zero Optimizer
zero_opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(zero_cost)

# 2 One weights
one_hidden_layer_1 = tf.add(tf.matmul(x_flat, one_weights_1), bias_weights_1)
one_hidden_layer_1 = tf.nn.relu(one_hidden_layer_1)

one_hidden_layer_2 = tf.add(tf.matmul(one_hidden_layer_1, one_weights_2), bias_weights_2)
one_hidden_layer_2 = tf.nn.relu(one_hidden_layer_2)

one_logits = tf.add(tf.matmul(one_hidden_layer_2, one_weights_3), bias_weights_3)

# One Cost
one_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= one_logits, labels= y))

# One Optimizer
one_opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(one_cost)

# 3 Best Practice (random) weights
bp_hidden_layer_1 = tf.add(tf.matmul(x_flat, bp_weights_1), bias_weights_1)
bp_hidden_layer_1 = tf.nn.relu(bp_hidden_layer_1)

bp_hidden_layer_2 = tf.add(tf.matmul(bp_hidden_layer_1, bp_weights_2), bias_weights_2)
bp_hidden_layer_2 = tf.nn.relu(bp_hidden_layer_2)

bp_logits = tf.add(tf.matmul(bp_hidden_layer_2, bp_weights_3), bias_weights_3)

# One Cost
bp_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= bp_logits, labels= y))

# One Optimizer
bp_opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(bp_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #Training
    for epoch in range(training_epochs):
        total_batches = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(zero_opt, feed_dict= {x: batch_x, y: batch_y})
            sess.run(one_opt, feed_dict= {x: batch_x, y: batch_y})
            sess.run(bp_opt, feed_dict= {x: batch_x, y: batch_y})
        # Log for the costs
        if epoch % display_step == 0:
            zero_cost_result = sess.run(zero_cost, feed_dict={x: batch_x, y: batch_y})
            one_cost_result = sess.run(one_cost, feed_dict={x: batch_x, y: batch_y})
            bp_cost_result = sess.run(bp_cost, feed_dict={x: batch_x, y: batch_y})
            print("Zero Cost Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(zero_cost_result))
            print("Opt Cost Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(one_cost_result))
            print("BP Cost Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(bp_cost_result))
        
    print('Optimisation Finished')
    # Test Model - Zero weights
    zero_correct_prediction = tf.equal(tf.argmax(zero_logits,1), tf.argmax(y,1))
    # Calculate accuracy
    zero_accuracy = tf.reduce_mean(tf.cast(zero_correct_prediction, "float"))
    # Test_size 
    test_size = 256
    print("Zero Accuracy:", zero_accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))

    # Test Model - One weights
    one_correct_prediction = tf.equal(tf.argmax(one_logits,1), tf.argmax(y,1))
    # Calculate accuracy
    one_accuracy = tf.reduce_mean(tf.cast(one_correct_prediction, "float"))
    # Test_size 
    test_size = 256
    print("One Accuracy:", one_accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))

    # Test Model - bp weights
    bp_correct_prediction = tf.equal(tf.argmax(bp_logits,1), tf.argmax(y,1))
    # Calculate accuracy
    bp_accuracy = tf.reduce_mean(tf.cast(bp_correct_prediction, "float"))
    # Test_size 
    test_size = 256
    print("Zero Accuracy:", bp_accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))
