import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Hyperparameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128

display_step = 1

n_input = 784 # Number of pixels
n_classes = 10 # Number of classes

n_hidden_layer = 256 # layer number of features

# Store weights and bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
bias = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#tf Graph input (placeholders)
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

# reshaping to get a linear 784 long input
x_flat = tf.reshape(x, [-1, n_input]) 

# Hidden Layer
layer_1 = tf.add(tf.matmul(x_flat,weights['hidden_layer']), bias['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1,weights['out']) , bias['out'])

#cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

#Optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(cost)

#Initilaization of variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training Cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Optimization
            sess.run(opt, feed_dict = {x: batch_x, y: batch_y})
        # Logs
        if epoch % display_step ==0:
            c = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
    print("Optimization Finished!")

    # Test Model
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Test_size 
    test_size = 256
    print("Accuracy:", accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))