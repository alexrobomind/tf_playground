import tensorflow as tf

import tensorflow.keras
#import tensorflow.keras.layers
import tensorflow.contrib.rnn

import tensorflow.summary

import numpy as np
import datetime
	
class NetBlock(tf.keras.layers.Layer):
	def __init__(self, units, skip_units, kernel_size, dilation = 1, **kwargs):
		super().__init__(**kwargs);
		
		self.units = units;
		self.kernel_size = kernel_size;
		self.dilation = dilation;
		self.skip_units = skip_units;
		
	def build(self, input_shape):
		self.conv_blocks = (
			tf.keras.layers.Conv1D(self.units, self.kernel_size, activation = 'sigmoid', name='conv_sigmoid', padding = 'same'),
			tf.keras.layers.Conv1D(self.units, self.kernel_size, activation = 'tanh', name = 'conv_tanh', padding = 'same')
		);
		
		if input_shape[2] != self.units:
			self.residual_projector = tf.keras.layers.Conv1D(self.units, 1, name = 'project_residual');
		else:
			self.residual_projector = None;
		
		if self.skip_units != self.units:
			self.skip_projector = tf.keras.layers.Conv1D(self.skip_units, 1, name = 'project_skip');
		else:
			self.skip_projector = None;
			
	def call(self, input):
		conv_out = self.conv_blocks[0](input) * self.conv_blocks[1](input);
		
		if self.residual_projector:
			residual = self.residual_projector(input);
		else:
			residual = input;
		
		if self.skip_projector:
			skip = self.skip_projector(conv_out);
		else:
			skip = conv_out;
		
		return (conv_out + residual, skip);

class Net(tf.keras.layers.Layer):
	def __init__(self, n_skip, blocks, **kwargs):
		super().__init__(**kwargs);
		
		self.skip = n_skip;
		self.block_configs = blocks;
	
	def build(self, input_shape):
		self.blocks = [NetBlock(skip_units = self.skip, **config) for config in self.block_configs];
	
	def call(self, input):
		batch_size = input.shape[0].value;
		sequence_length = input.shape[1].value;
		
		skip = tf.zeros(shape = (batch_size, sequence_length, self.skip), dtype = tf.float32);
		
		for block in self.blocks:
			(input, skipdelta) = block(input);
			
			skip = skip + skipdelta;
		
		return skip;