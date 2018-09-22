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
		'outputs': 16,
		'hidden': [{'filters': 16, 'kernel_size': 2, 'dilation_rate': 2**i}]
	} for i in range(0, 12)];
	
	for i in range(0, 1):
		input = wavenet.Wavenet(blocks, 16)(input, condition);
	
	for i in range(0, 2):
		input = tf.keras.layers.Conv1D({
			'filters': 16,
			'kernel_size': 1,
			'activation': 'relu'
		})(input);
	
	return input;

constant_size = 10;

inputs = tf.placeholder(shape = [None, None, 1], name = 'inputs');
constants = tf.placeholder(shape = [10], name = 'constants');
outputs = tf.identity(model(inputs), 'outputs');


#output = model(tf.placeholder(shape = [1, 2049, 1], name = 'input', dtype=tf.float32), tf.placeholder(shape = [4], name = 'condition_constants',dtype=tf.float32));
#output = tf.identity(output, 'output');

tf.train.export_meta_graph(filename='graph');
tf.reset_default_graph();
#tf.train.import_meta_graph('graph', input_map = {
#	'input' : tf.zeros(shape = [1, 2049, 1], name = 'input_rep'),
#	'condition_constants' : tf.zeros(shape = [4], name = 'condition_rep')
#});

output = tf.get_default_graph().get_tensor_by_name('output:0');

init = tf.global_variables_initializer();

with tf.Session() as sess:
	timestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
	writer = tf.summary.FileWriter("logdir/{}".format(timestring), sess.graph);
	
	sess.run(init);
	sess.run(output);
	writer.close();