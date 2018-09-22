import tensorflow as tf

import tensorflow.data
import tensorflow.train
import tensorflow.python_io

import os.path as path
import os

import scipy.io as sio
import scipy.io.wavfile

import glob

import numpy as np

class ConvResnetBlock(tf.keras.layers.Layer):
	def __init__(
		self,
		name = 'wavenet_block',
		outputs = 16,
		hidden = [{'filters': 32, 'kernel_size': 2}],
		skip = 16,
		causal = True,
		**kwargs
	):
		if causal:
			hidden = [{**e, 'padding': 'valid'} for e in hidden];
		
		super().__init__(name = name, **kwargs);
		
		self.outputs = outputs;
		self.skip = skip;
		self.hidden = hidden;
		self.causal = causal;
		
	
	def build(self, input_shape):
		self.conv_1x1_skip = tf.keras.layers.Conv1D(
			filters = self.skip,
			kernel_size = 1,
			name = '1x1_skip'
		);
		
		self.conv_1x1_out = tf.keras.layers.Conv1D(
			filters = self.outputs,
			kernel_size = 1,
			name = '1x1_out'
		);
		
		self.cconv_hidden = [
			(
				tf.keras.layers.Conv1D(
					activation = 'tanh',
					name = 'act_conv',
					**config
				),
				tf.keras.layers.Conv1D(
					activation = 'sigmoid',
					name = 'gate_conv',
					**config
				)
			)
			for config in self.hidden
		];
	
	def call(self, input, constants = None):
		intermediate = input;
		
		if(constants is not None):
			intermediate = tf.concat([intermediate, constants], axis = 2);
		
		for act, gate in self.cconv_hidden:
			next = act(intermediate) * gate(intermediate);
			
			if self.causal:
				next = tf.pad(next, [[0, 0], [intermediate.shape[1].value - next.shape[1].value, 0], [0, 0]]);
			
			intermediate = next;
		
		skip = self.conv_1x1_skip(intermediate);
		out = self.conv_1x1_out(intermediate);
		
		# If possible, wire up a residual connection
		if(out.shape == input.shape):
			out = out + input;
		
		return (skip, out);

class ConvResnet(tf.keras.layers.Layer):
	def __init__(self, blocks, outputs, **kwargs):
		super().__init__(**kwargs);
		self.outputs = outputs;
		self.block_configs = [{**block, 'skip': outputs} for block in blocks];
	
	def build(self, input_shape):
		self.blocks = [ConvResnetBlock(**config) for config in self.block_configs];
	
	def call(self, input, constants = None):
		input_shape = tf.shape(input);
		constant_shape = tf.shape(constants);
		
		if constants is not None:
			constants = tf.broadcast_to(
				constants,
				shape = [input_shape[0], input_shape[1], constant_shape[-1]],
				name = 'conditioning'
			);
		
		output = tf.zeros(shape = [input_shape[0], input_shape[1], self.outputs]);
		
		for block in self.blocks:				
			skip, input = block(input, constants);
			output = output + skip;
		
		return output;

class Wavenet(ConvResnet):
	def __init__(self, blocks, outputs, **kwargs):
		super().__init__([{**block, 'causal' : True} for block in blocks], outputs, **kwargs);