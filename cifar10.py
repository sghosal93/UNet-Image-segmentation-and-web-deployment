from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")
tf.app.flags.DEFINE_float('epsilon', 1e-5, """ Small epsilon to prevent division by zero """)

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

def segmentation_distorted_inputs(rawImageFns, rawLabelMasks, num_examples_per_epoch, shuffle, _condition, mode):
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	names, images, labels = cifar10_input.segmentation_distorted_inputs(imageFns=rawImageFns, labelMasks=rawLabelMasks, batch_size=FLAGS.batch_size, num_examples_per_epoch=num_examples_per_epoch, shuffle=shuffle, _condition=_condition, mode=mode)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return names, images, labels

def encoder_layer(layer_input=None, n_filters=None, k_size=3, stride=2, activation=None, padding='SAME', trainable=True, name=None):
	if n_filters == None:
		print('Number of filter not provided to encoder layer {}'.format(name))
		return

	with tf.name_scope('name'):
		mean, variance = tf.nn.moments(layer_input, axes=[0, 1, 2], name='conv_batch_norm', keep_dims=True)
		normalized = tf.divide(tf.subtract(layer_input, mean), tf.sqrt(variance + FLAGS.epsilon))
		
		out = tf.layers.conv2d(inputs=normalized, filters=n_filters, kernel_size=k_size, strides=1, padding='SAME', activation=None, use_bias=True, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), bias_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), trainable=trainable, name='{}_{}'.format(name, 'conv2d'))
	
		if activation == tf.nn.relu:
			out = tf.nn.relu(out, name='{}_{}'.format(name, 'relu'))
		elif activation == tf.sigmoid:	
			out = tf.sigmoid(out, name='{}_{}'.format(name, 'sigmoid'))
		elif activation == tf.nn.softmax:
			out = tf.nn.softmax(out, name='{}_{}'.format(name, 'softmax'))	
		
		out = tf.layers.max_pooling2d(inputs=out, pool_size=stride, strides=stride, padding='SAME', name='{}_{}'.format(name, 'max_pooling2d'))
		
		return out

def flatten(layer_input, name):
	with tf.name_scope(name):
		sh = layer_input.get_shape().as_list()[1:]
		return tf.reshape(layer_input, shape=[-1, sh[0]*sh[1]*sh[2]])

def add_layers(a, b):
	return tf.concat([a, b], axis=-1)

def decoder_layer(layer_input=None, is_flat=False, last_img_shape=None, n_filters=None, k_size=3, stride=2, activation=None, padding='SAME', unpooling_size=None, trainable=True, name=None):
	if n_filters == None:
		print('Number of filter not provided to decoder layer {}'.format(name))
		return

	if is_flat == True:
		sh2 = last_img_shape[1:]
		layer_input = tf.reshape(layer_input, shape=[-1, sh2[0], sh2[1], sh2[2]])
	
	mean, variance = tf.nn.moments(layer_input, axes=[0, 1, 2], name='conv_batch_norm', keep_dims=True)
	normalized = tf.divide(tf.subtract(layer_input, mean), tf.sqrt(variance + FLAGS.epsilon))
	
	out = tf.image.resize_images(normalized, size=unpooling_size)

	out = tf.layers.conv2d_transpose(inputs=out, filters=n_filters, kernel_size=k_size, strides=1, padding='SAME', activation=None, use_bias=True, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), bias_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), trainable=trainable, name='{}_{}'.format(name, 'conv2d'))

	if activation == tf.nn.relu:
		out = tf.nn.relu(out, name='{}_{}'.format(name, 'relu'))
	elif activation == tf.nn.sigmoid:	
		out = tf.nn.sigmoid(out, name='{}_{}'.format(name, 'sigmoid'))
	elif activation == tf.nn.softmax:
		out = tf.nn.softmax(out, name='{}_{}'.format(name, 'softmax'))
	
	return out
