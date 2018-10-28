import tensorflow as tf
import tensorflow_probability as tfp

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

def _validate_shape(coeffs):
	coeffs_shape = coeffs.shape.as_list();
	coeffs_shape[-1] = 3;
	coeffs.set_shape(coeffs_shape);

# Samples a multi-logistic distribution from N basic distributions given a tensor of shape [..., N, 3]
def distribution(coeffs, name = "MixedLogistic"):
		_validate_shape(coeffs);
		
		distribution = tfp.distributions.MixtureSameFamily(
			mixture_distribution = tfp.distributions.Categorical(logits = coeffs[..., :, 0], validate_args=True),
			components_distribution = tfp.distributions.Logistic(loc = coeffs[..., :, 1], scale = tf.exp(coeffs[..., :, 2]), validate_args=True)			
		);
		
		return distribution;

if __name__ == "__main__":
	coeffs = tf.tile(
		[
			[
				[0.5, 0.0, 0.2], [0.25, 50.0, 10.0], [0.25, -50.0, 10.0]
			]
		],
		multiples = [10000000, 1, 1],
		name = "Input"
	);

	coeffs = tf.identity(coeffs);
	dist = distribution(coeffs);
	samples = dist.sample();
	samples = tf.identity(samples, name = 'output');
	tf.summary.histogram('Output', samples);
	tf.summary.scalar('KLDiv', tfp.distributions.kl_divergence(dist, dist));

	summaries = tf.summary.merge_all();

	with tf.Session() as sess:
		timestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
		writer = tf.summary.FileWriter("logdir/{}".format(timestring), sess.graph);
		
		for i in range(0, 1):
			writer.add_summary(sess.run(summaries), i);
		
		writer.close();