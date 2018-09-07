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
		dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
		templates = [{
			'units' : 32,
			'kernel_size' : 3
			}, {
			'units' : 64,
			'kernel_size' : 3
			#}, {
			#'units' : 125,
			#'kernel_size' : 3
		}];

		configs = [];
		for template in templates:
			for dilation in dilations:
				configs = configs + [{
					'dilation' : dilation,
					**template
				}];				

		self.net = ns_net.Net(32, configs);
		
		dense_units = [32, 32];
		self.post = [tf.keras.layers.Conv1D(n, kernel_size = 1, name = 'post_1x1', activation='relu') for n in dense_units];
		self.final = tf.keras.layers.Conv1D(self.units, kernel_size = 1, name = 'final_1x1');
	
	def call(self, input):		
		output = self.net(input);
		
		for post in self.post:
			output = post(output) + output;
		
		return self.final(output);

with tf.name_scope('datasource'):
	src = synthetic_data.langmuir_data(16);

input_ids = ['bias', 'current'];
model_input = tf.stack([src[i] for i in input_ids], axis = 2);
#model_input = tf.check_numerics(model_input, message = 'Numerical error in model input');

references = [src[i] for i in src if i is not 't'];
references = [tf.broadcast_to(v, shape = src['bias'].shape) for v in references];
reference = tf.stack(references, axis = 2);

model = Model(reference.shape[2].value);
output = model(model_input);

loss = tf.losses.mean_squared_error(
	output, reference
);

optimizer = tf.train.AdamOptimizer(0.01);
train = optimizer.minimize(loss);

# Summaries
tf.summary.scalar('loss', loss);

keys = [k for k in src if k is not 't'];
for i in range(0, len(keys)):
	tf.summary.scalar(
		'loss_{}'.format(keys[i]),
		tf.losses.mean_squared_error(
			output[:, :, i], reference[:, :, i]
		)
	);

init = tf.global_variables_initializer();
summarize = tf.summary.merge_all();

with tf.Session() as session:
	session.run([init]);
	
	# This class saves all data to a subdirectory of "output" with the current datetime
	writer = tf.summary.FileWriter(
		datetime.datetime.now().strftime("output/%I_%M%p on %B %d %Y"),
		session.graph
	);
	
	for i in range(0, 10000):
		loss_val, _, summary, output_val = session.run([loss, train, summarize, output]);
		#print('Out: {}'.format(output_val));
		print('Loss: {}'.format(loss_val));
		writer.add_summary(summary, i);
		
	writer.close();