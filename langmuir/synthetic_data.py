import tensorflow as tf
import numpy as np

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

# Generates langmuir U-I traces with slightly randomized characteristics
def langmuir_u_i(input, electron_temperatures, electron_densities, ion_saturation_currents, plasma_potentials):
	batch_size = input.shape[0];
	
	ion_contribution = 
	
	electron_saturation = random_uniform(shape = (batch_size,), minval = 1, maxval = 5);
	electron_saturation_slope = random_uniform(shape = (batch_size,), minval = 0, maxval = 0.01);
	
	

def langmuir_trace(frequency, n_samples, batch_size):
	#frequencies = random_uniform(shape = (batch_size), minval = 30, maxval = );
	#phases      = random_