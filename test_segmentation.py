from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2, time, glob, json, argparse

import numpy as np
import tensorflow as tf
import cifar10_input
import cifar10
from copy import deepcopy

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

region = 'combined'
modality = 'whitelight'
condition = 'gingivitis'
savedist = 'white_cmb'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../Trained_Models/%s_%s_train_%s__Default_No_Balance' % (condition, modality, region), """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../Trained_Models/%s_%s_train_%s__Default_No_Balance' % (condition, modality, region), """Location of saved model""")

tf.set_random_seed(1234)

def unet_Model(model_input):
	FILTERS = [16, 32, 64, 128, 256]
	NUM_NN_IMAGE_OUTPUTS = 1

	# Encoder
	encode1 = cifar10.encoder_layer(layer_input=model_input, n_filters=FILTERS[0], activation=tf.nn.relu, trainable=True, name='Encoder__L1')
	encode2 = cifar10.encoder_layer(layer_input=encode1, n_filters=FILTERS[1], activation=tf.nn.relu, trainable=True, name='Encoder__L2')
	encode3 = cifar10.encoder_layer(layer_input=encode2, n_filters=FILTERS[2], activation=tf.nn.relu, trainable=True, name='Encoder__L3')
	encode4 = cifar10.encoder_layer(layer_input=encode3, n_filters=FILTERS[3], activation=tf.nn.relu, trainable=True, name='Encoder__L4')
	encode5 = cifar10.encoder_layer(layer_input=encode4, n_filters=FILTERS[4], activation=tf.nn.relu, trainable=True, name='Encoder__L5')
	
	# Flatten
	flat = cifar10.flatten(encode5, 'Flatten')
	
	# Decoder
	decode1 = cifar10.decoder_layer(layer_input=flat, is_flat=True, last_img_shape=encode5.get_shape().as_list(), n_filters=FILTERS[3], activation=tf.nn.relu, unpooling_size=[30, 40], trainable=True, name='Decoder__L1')
	decode1 = cifar10.add_layers(decode1, encode4)		
	decode2 = cifar10.decoder_layer(layer_input=decode1, is_flat=False, last_img_shape=None, n_filters=FILTERS[2], activation=tf.nn.relu, unpooling_size=[60, 80], trainable=True, name='Decoder__L2')
	decode2 = cifar10.add_layers(decode2, encode3)
	decode3 = cifar10.decoder_layer(layer_input=decode2, is_flat=False, last_img_shape=None, n_filters=FILTERS[1], activation=tf.nn.relu, unpooling_size=[120, 160], trainable=True, name='Decoder__L3')
	decode3 = cifar10.add_layers(decode3, encode2)
	decode4 = cifar10.decoder_layer(layer_input=decode3, is_flat=False, last_img_shape=None, n_filters=FILTERS[0], activation=tf.nn.relu, unpooling_size=[240, 320], trainable=True, name='Decoder__L4')
	decode4 = cifar10.add_layers(decode4, encode1)
	decode5 = cifar10.decoder_layer(layer_input=decode4, is_flat=False, last_img_shape=None, n_filters=NUM_NN_IMAGE_OUTPUTS, activation=tf.nn.sigmoid, unpooling_size=[480, 640], trainable=True, name='Decoder__L5')
	
	return decode5

def inference_batch(rawImage, rawLabelFns, folder):
	num_examples_per_epoch = len(rawImage)
	max_steps = len(rawImage)

	print('Number of test images: %d' % len(rawImage))

	if not os.path.exists('static'):
		os.mkdir('static')

	des_folder = 'static'

	with tf.Graph().as_default():
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		names, images, labels = cifar10.segmentation_distorted_inputs(rawImage, rawLabelFns, num_examples_per_epoch, shuffle=True, _condition=str(condition), mode="test")
		batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([names, images, labels], capacity=2*FLAGS.num_gpus)
		names_batch, image_batch, label_batch = batch_queue.dequeue()

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			network_output = tf.squeeze(unet_Model(image_batch), 3)

			saver = tf.train.Saver()
			ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				print('No checkpoint file found')
				return

			while True:
				names_b, image_b, label_b, net_out = sess.run([names_batch, image_batch, label_batch, network_output])

				for i in range(len(names_b)):
					name = (names_b[i]).decode("utf-8")
					img = image_b[i, :, :, :]
					gnd = np.squeeze(label_b[i, :, :])
					gnd = np.array(gnd, dtype=np.int32)
					pred = np.array(net_out[i, :, :], dtype=np.int32)

					# Save image
					patient = str(name.split('/')[-2])
					fname = name[name.rfind('/')+1:name.rfind('.')]
					
					orig_im = cv2.imread(name) 
					cv2.imwrite(des_folder+'/%s_in.png'%(fname), orig_im)
					cv2.imwrite(des_folder+'/%s_pred.png'%(fname), 255*pred)
					cv2.imwrite(des_folder+'/%s_out.png'%(fname), overlay(deepcopy(orig_im), pred, color=[0, 255, 0]))
					cv2.imwrite('/home/user1/Desktop/3_Live_segmentation_outputs'+'/%s_out.png'%(fname), overlay(deepcopy(orig_im), pred, color=[0, 255, 0]))
					
					marked_gingivitis_pixels = int(len(pred[pred==1]))
					unmarked_pixels = int(len(pred[pred==0]))
					percentage_white = 100*marked_gingivitis_pixels/(marked_gingivitis_pixels + unmarked_pixels)

					Gingivitis_pixel_dist = {}
					Gingivitis_pixel_dist['Pixels_and_percentage'] = {'Gingivitis_Pixels':marked_gingivitis_pixels, 'Percentage_wrt_image':round(percentage_white, 2)}

					savename = '/home/user1/Desktop/2_Live_segmentation_pixel_results/%s_pix.json'%(fname)
	
					with open(savename, 'w') as f:
						json.dump(Gingivitis_pixel_dist, f)

					TOTAL_FILES = glob.glob(des_folder + "/")
					if ".DS_Store" in TOTAL_FILES:
						TOTAL_FILES.remove(".DS_Store")

					if len(TOTAL_FILES) == max_steps:
						return

			coord.request_stop()
			coord.join(threads)

def overlay(bckgndImg, _mask, color=None):
	_bckgndImg = deepcopy(bckgndImg)
	color = np.array(color)
	shpY, shpX, _ = _bckgndImg.shape
	for j in range(shpY): # height
		for i in range(shpX): # width
			if _mask[j, i] == 1:
				_bckgndImg[j, i, :] = color
	return _bckgndImg

def main(image_path):

	print("Starting evaluation..")

	inference_batch(image_path, image_path, folder="test")

if __name__=='__main__':
	tf.app.run()
