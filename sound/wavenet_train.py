import tensorflow as tf

import tensorflow.data
import tensorflow.train
import tensorflow.python_io
import tensorflow.summary

import os.path as path
import os

import scipy.io as sio
import scipy.io.wavfile

import glob

import numpy as np

import datetime

import wavenet

def model(input, condition = None):
	blocks = [{
		'outputs': 512,
		'hidden': [{'filters': 256, 'kernel_size': 2, 'dilation_rate': 2**i}]
	} for i in range(0, 10)];
	
	for i in range(0, 1):
		input = wavenet.Wavenet(blocks, 256)(input, condition);
	
	for i in range(0, 2):
		input = tf.keras.layers.Conv1D(
			filters = 256,
			kernel_size = 1,
			activation = 'relu'
		)(input);
		
	n_distributions = 10;
	
	def output_layer():
		return tf.keras.layers.Conv1D(
			filters = n_distributions,
			kernel_size = 1
		)(input);
	
	weights = output_layer();
	means = output_layer();
	widths = output_layer();
	
	return (weights, means, widths);

constant_size = 10;

inputs = tf.placeholder(shape = [None, None, 1], name = 'inputs', dtype=tf.float32);
constants = tf.placeholder(shape = [10], name = 'constants', dtype=tf.float32);

(weights, means, widths) = model(inputs, constants);

weights = tf.identity(weights, name = 'output_weights');
means = tf.identity(means, name = 'output_means');
widths = tf.identity(widths, name = 'output_widths');


#output = model(tf.placeholder(shape = [1, 2049, 1], name = 'input', dtype=tf.float32), tf.placeholder(shape = [4], name = 'condition_constants',dtype=tf.float32));
#output = tf.identity(output, 'output');

tf.train.export_meta_graph(filename='graph');
tf.reset_default_graph();
tf.train.import_meta_graph('graph', input_map = {
	'inputs' : tf.zeros(shape = [1, 2049, 1], name = 'input_rep'),
	'constants' : tf.zeros(shape = [10], name = 'condition_rep')
});

outputs = tf.get_default_graph().get_tensor_by_name('output_weights:0');

init = tf.global_variables_initializer();

with tf.Session() as sess:
	timestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
	writer = tf.summary.FileWriter("logdir/{}".format(timestring), sess.graph);
	
	sess.run(init);
	#sess.run(outputs);
	writer.close();