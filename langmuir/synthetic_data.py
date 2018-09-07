import tensorflow as tf
import numpy as np
import math

import tensorflow.keras
import tensorflow.keras.layers

import tensorflow.train
import tensorflow.losses

import tensorflow.data
import tensorflow.distributions

import tensorflow.summary
import datetime

import tensorflow.contrib.rnn
import tensorflow.nn

import tensorflow.initializers

import tensorflow.keras.preprocessing.text as kpt

import matplotlib.pyplot as plt

# Generates langmuir U-I traces with slightly randomized characteristics
def langmuir_u_i(bias, t_e, i_sat, v_plasma, m_i_over_m_e):
	batch_size = bias.shape[0];
	sequence_length = bias.shape[1];
	
	electron_charge = tf.constant(1.0, name = 'q_e');
	k_b = tf.constant(1.0, name = 'k_b');
	
	def random_per_run(min, max, name):
		with tf.name_scope(name):
			min = tf.identity(float(min), name = 'min');
			max = tf.identity(float(max), name = 'max');
			shape = tf.TensorShape([tf.Dimension(batch_size), tf.Dimension(1)]);
			tmp = tf.random_uniform(shape = shape, minval = min, maxval = max);
		
		return tmp;
	
	
	with tf.name_scope('electron_current'):
		e_e = k_b * t_e;
		inv_2pi = tf.constant(1 / 2 * math.pi, name = 'one_by_2_pi');
		m_i_over_m_e = tf.identity(m_i_over_m_e, name = 'ion_to_electron_mass_ratio');
		
		electron_contribution = (
			i_sat
			* tf.sqrt(m_i_over_m_e * inv_2pi)
			* tf.exp(electron_charge * (bias - v_plasma) / e_e)
		);
	
	with tf.name_scope('electron_limit'):
		electron_limit = (
			random_per_run(min = 5, max = 20, name = 'e_sat_c0_random')
			+ bias * random_per_run(min = 0, max = 0.01, name = 'e_sat_c1_random')
		) * i_sat;
	
	with tf.name_scope('softmin'):
		electron_contribution = -tf.log(
			tf.exp(-electron_contribution) +
			tf.exp(-electron_limit)
		);
		electron_contribution = tf.maximum(electron_contribution, 0);
	
	result = electron_contribution - i_sat;
	
	return result;

def synthetic_probe(t_e, i_sat, v_plasma, mass_ratio, n, dt):
	dt = tf.identity(dt, name = 'dt');
	
	with tf.name_scope('t'):
		t = tf.reshape(
			tf.range(0, n, dtype = tf.float32),
			shape = (1, -1)
		) * dt;
	
	with tf.name_scope('random_frequency'):
		frequency = tf.random_uniform(
			shape = (t_e.shape[0].value, 1),
			minval = 300,
			maxval = 5000
		);
	
	with tf.name_scope('bias'):
		with tf.name_scope('random_amplitude'):
			amplitude = tf.random_uniform(
				shape = (t_e.shape[0].value, 1),
				minval = 200,
				maxval = 500
			);
		
		with tf.name_scope('random_offset'):
			offset = tf.random_uniform(
				shape = (t_e.shape[0].value, 1),
				minval = -50,
				maxval = 50
			);
			
		bias = tf.sin(frequency * t) * amplitude + offset;
		
	with tf.name_scope('u_i_curve'):
		current = langmuir_u_i(bias, t_e, i_sat, v_plasma, mass_ratio);
		
	noise_level = tf.constant(2.0, name = 'noise_level');
	
	with tf.name_scope('noise'):
		noise = tf.random_uniform(
			current.shape,
			minval = -noise_level,
			maxval = noise_level
		);
	
	output = current + noise;
	
	return (t, bias, output);
	
	
#test = lambda name : tf.ones(shape = (1, 2000), name = name);

def langmuir_data(batch_size):
	def random_range(name, min, max):
		shape = [batch_size, 1];
		with tf.name_scope(name):
			return tf.random_uniform(
				shape = shape,
				minval = min,
				maxval = max
			);
			
	t_e  = random_range('t_e', 5, 200);
	i_sat = random_range('i_sat', 0.2, 5);
	v_plasma = random_range('v_p', -50, 150);
	mass_ratio = 1.0;

	(t, bias, current) = synthetic_probe(t_e, i_sat, v_plasma, mass_ratio, 20000, 1e-5);
	
	return {
		't' : t,
		't_e' : t_e,
		'i_sat' : i_sat,
		'v_p' : v_plasma,
		'bias' : bias,
		'current' : current
	};

#with tf.name_scope('input_data'):
#	data = langmuir_data(1);
#
#with tf.Session() as session:
#	# This class saves all data to a subdirectory of "output" with the current datetime
#	writer = tf.summary.FileWriter(
#		datetime.datetime.now().strftime("output/%I_%M%p on %B %d %Y"),
#		session.graph
#	);
#	writer.close();
#	
#	for i in range(0, 100):
#		data_out = session.run(data);
#		plt.plot(data_out['bias'][0,:], data_out['current'][0,:]);
#		plt.show();
#		plt.plot(data_out['t'][0,:], data_out['current'][0,:]);
#		plt.show();
#plt.show();