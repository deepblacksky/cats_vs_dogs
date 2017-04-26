import tensorflow as tf


def inference(image, batch_size, n_classes):
    """
    build model
    :param image:image batch, 4-D tensor, [batch_size, width, height, 3], dtype = tf.float32 
    :param batch_size: batch size
    :param n_classes: num of class, it is two about cat and dog
    :return: compute logits, tensor, [batch_size], tf.int32 
    """

    # conv1 shape = [kernel_size, kernel_size, channel, kernel_number]
    with tf.variable_scope('conv1') as scope:
        weight1 = tf.get_variable(name='weight1', shape=[3, 3, 3, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        bias1 = tf.get_variable(name='bias1', shape=[16], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(image, weight1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bias=bias1), name=scope.name)

    # pool1 and norm1
    with tf.variable_scope('pool1_and_norm1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weight2 = tf.get_variable(name='weight2', shape=[3, 3, 16, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        bias2 = tf.get_variable(name='bias2', shape=[16], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(norm1, weight2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2), name=scope.name)

    # pool2 and norm2
    with tf.variable_scope('pool2_and_norm2') as scope:
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling2')

    # local3 fcl 128 node
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight3 = tf.get_variable('weight3', shape=[dim, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        bias3 = tf.get_variable('bias3', shape=[128], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3, name=scope.name)

    # local4 fcl 128 node
    with tf.variable_scope('local4') as scope:
        weight4 = tf.get_variable(name='weight4', shape=[128, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        bias4 = tf.get_variable(name='bias4', shape=[128], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4, name=scope.name)

    # softmax5
    with tf.variable_scope('softmax_linear') as scope:
        weight5 = tf.get_variable(name='weight5', shape=[128, n_classes], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        bias5 = tf.get_variable(name='bias5', shape=[n_classes], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(local4, weight5), bias5, name=scope.name)

    return logits


def losses(logits, labels):
    """
    compute loss using logits and label
    :param logits: tensor, output inference()
    :param labels: label tensor
    :return: loss tensor
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(name=scope.name+'/loss', tensor=loss)
        return loss


def training(loss, learning_rate):
    """
    Training optimizes, sess.run() need this function return
    :param loss: loss tensor from losses()
    :param learning_rate: learning rate
    :return: train op
    """
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, label):
    """
    Evaluate the quality of the logits at predicting the label.
    :param logits: 
    :param label: 
    :return:  A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, label, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy

