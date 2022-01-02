from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import pandas as pd
import random
import json

IMAGE_SIZE = 24
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2046
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 512

NUM_PREPROCESS_THREADS = 8

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

DOWNSIZE_FACTOR = 1.0

random.seed(1337)

def readCSV(fn):

	csv = pd.read_csv(fn, keep_default_na=False, na_values=['NaN'])

	df_test = csv

	imagePaths_test = list(df_test['image_dir'])

	pathsNumpy_test = np.array(imagePaths_test)
	labelFnsNumpy_test = pathsNumpy_test

	return pathsNumpy_test, labelFnsNumpy_test

def segmentation_readImages(input_queue, mode):

	file_contents = tf.read_file(input_queue[0])
	label_file_contents = tf.read_file(input_queue[1])
	example = tf.cast(tf.image.decode_image(file_contents, channels=3), tf.float32)
	labelMask = tf.cast(tf.image.decode_image(label_file_contents, channels=3), tf.float32)

	if mode == "train":
		chance_lr = tf.random_normal([1])  
		chance_ud = tf.random_normal([1]) 

		if tf.reduce_all(tf.greater(chance_lr, tf.Variable(1.0))) is True:
			example = tf.image.flip_left_right(example)
			labelMask = tf.image.flip_left_right(labelMask)

		if tf.reduce_all(tf.greater(chance_ud, tf.Variable(1.0))) is True:
			example = tf.image.flip_up_down(example)
			labelMask = tf.image.flip_up_down(labelMask)

		example = tf.image.random_brightness(example, 10.0)

	example.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
	labelMask.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
	return input_queue[0], example, labelMask

def _segmentation_generate_image_and_label_batch(_name, image, label, min_queue_examples, batch_size, shuffle):
	num_preprocess_threads = NUM_PREPROCESS_THREADS

	if shuffle:
		n, images, label_batch = tf.train.shuffle_batch([_name, image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
		return n, images, label_batch
	else:
		n, images, label_batch = tf.train.batch([_name, image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size)

		return n, images, label_batch

def segmentation_distorted_inputs(imageFns, labelMasks, batch_size, num_examples_per_epoch, shuffle, _condition, mode):
	imageFnsTensor = tf.convert_to_tensor(imageFns, dtype=tf.string)
	labelMasksTensor = tf.convert_to_tensor(labelMasks, dtype=tf.string)
	inputQueueTrain = tf.train.slice_input_producer([imageFnsTensor, labelMasksTensor], shuffle=False)
	name, raw_image, labelMask = segmentation_readImages(inputQueueTrain, mode)
	float_image = tf.image.per_image_standardization(raw_image)

	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
	print ('Filling queue with %d images before starting to train. This will take a few minutes.' % min_queue_examples)

	_name, _image, _mask = _segmentation_generate_image_and_label_batch(name, float_image, labelMask, min_queue_examples, batch_size, shuffle=shuffle)
	
	images = tf.image.resize_bicubic(_image, tf.convert_to_tensor([int(480 / DOWNSIZE_FACTOR), int(640 / DOWNSIZE_FACTOR)], dtype=tf.int32))
	labelsRGB = tf.image.resize_bicubic(_mask, tf.convert_to_tensor([int(480 / DOWNSIZE_FACTOR), int(640 / DOWNSIZE_FACTOR)], dtype=tf.int32))

	labelsFullRange = tf.image.rgb_to_grayscale(labelsRGB)
	_max = tf.clip_by_value(tf.reduce_max(labelsFullRange), 1, 255)
	labels = tf.divide(labelsFullRange, _max)

	return _name, images, labels

