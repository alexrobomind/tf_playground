import tensorflow as tf

import tensorflow.keras
#import tensorflow.keras.layers
import tensorflow.contrib.rnn

import tensorflow.summary

import numpy as np
import datetime

import convnet_1 as ns_net
import synthetic_data

class Model(tf.keras.layers.Layer):
	def __init__(self, units, **kwargs):
		super().__init__(kwargs);
		
		self.units = units;

	def build(self, input_shape):
		dilations = [1, 2, 4];
		templates = [{
			'units' : 7,
			'kernel_size' : 3
			}, {
			'units' : 5,
			'kernel_size' : 3
		}];

		configs = [];
		for template in templates:
			for dilation in dilations:
				configs = configs + [{
					'dilation' : dilation,
					**template
				}];				

		self.net = ns_net.Net(7, configs);
		
		dense_units = [20, self.units];
		self.post = [tf.keras.layers.Conv1D(n, kernel_size = 1, name = 'post_1x1') for n in dense_units];
	
	def call(self, input):		
		output = self.net(input);
		
		for post in self.post:
			output = post(output);
		
		return output;

def datasource():
	batch_size = 64;
	return synthetic_data.langmuir_data(batch_size);

def loss(input, reference):
	return tf.losses.mean_squared_error(
		input, reference
	);

with tf.name_scope('datasource'):
	src = datasource();

input_ids = ['bias', 'current'];
model_input = tf.stack([src[i] for i in input_ids], axis = 2);

references = [src[i] for i in src if i is not 't'];
references = [tf.broadcast_to(v, shape = src['bias'].shape) for v in references];
reference = tf.stack(references, axis = 2);

model = Model(reference.shape[2].value);
output = model(model_input);

target = loss(output, reference);	

with tf.Session() as session:
	# This class saves all data to a subdirectory of "output" with the current datetime
	writer = tf.summary.FileWriter(
		datetime.datetime.now().strftime("output/%I_%M%p on %B %d %Y"),
		session.graph
	);
	writer.close();