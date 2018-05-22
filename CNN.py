import numpy as np
import tensorflow as tf
import config


class Network(object):
    def __init__(self, network_type):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None])
        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step, 2e3, 1e-4)
        
        if network_type == 'BP':
            self.logits = self.bp(self.X_inputs)
        elif network_type == 'RBF':
            self.logits = self.rbf(self.X_inputs)
        elif network_type == 'CNN':
            self.logits = self.cnn(self.X_inputs)
        
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.accuracy = self.get_accuracy(self.logits, self.labels)
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        
    def bp(self, X_inputs):
        layer = tf.contrib.layers.flatten(X_inputs)
        layer = tf.layers.dense(layer, config.hidden_num)
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        out = tf.layers.dense(layer, config.class_num)
        return out
    
    def rbf(self, X_inputs):
        layer = tf.contrib.layers.flatten(X_inputs)
        layer = tf.layers.dense(layer, config.hidden_num)
        c = 0
        layer = tf.exp(-config.rbf_gamma * np.abs(layer - c))
        out = tf.layers.dense(layer, config.class_num)
        return out

    def cnn(self, X_inputs):
        layer = tf.layers.conv2d(X_inputs, 32, 3, padding='same', activation=None)
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, self.keep_prob)
        layer = tf.layers.max_pooling2d(layer, 2, 2)
        
        layer = tf.layers.conv2d(X_inputs, 64, 3, padding='same', activation=None)
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, self.keep_prob)
        layer = tf.layers.max_pooling2d(layer, 2, 2)
        
        layer = tf.layers.conv2d(X_inputs, 128, 3, padding='same', activation=None)
        layer = tf.layers.batch_normalization(layer, training=self.training)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer, self.keep_prob)
        layer = tf.layers.max_pooling2d(layer, 2, 2)
        
        flat = tf.contrib.layers.flatten(layer)
        out = tf.layers.dense(flat, config.class_num)
        return out

    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate, config.beta1, config.beta2).minimize(loss, global_step=self.global_step)
        return loss, optimizer

    def get_accuracy(self, logits, labels):
        softmax_logits = tf.nn.softmax(logits)
        correct_pre = tf.equal(tf.argmax(softmax_logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
        return accuracy
