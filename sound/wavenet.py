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

class ConvResnetBlock(tf.keras.models.Model):
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
		#@tf.contrib.layers.recompute_grad
		def inner(input, constants):
			intermediate = input;
			
			if(constants is not None):
				intermediate = tf.concat([intermediate, constants], axis = 2);
			
			for act, gate in self.cconv_hidden:
				next = act(intermediate) * gate(intermediate);
				
				if self.causal:
					causal_padding = (act.kernel_size[0] - 1) * act.dilation_rate[0];
					next = tf.pad(next, [[0, 0], [causal_padding, 0], [0, 0]]);
				
				intermediate = next;
			
			skip = self.conv_1x1_skip(intermediate);
			out = self.conv_1x1_out(intermediate);
			
			# If possible, wire up a residual connection
			if(out.shape[2] == input.shape[2]):
				out = out + input;
			
			return (skip, out);
		
		return inner(input, constants);
	
	def padding(self):
		if not self.causal:
			return 0;
				
		paddings = [
			(act.kernel_size[0] - 1) * act.dilation_rate[0]
			for act, _ in self.cconv_hidden
		];
		
		return sum(paddings);

class ConvResnet(tf.keras.models.Model):
	def __init__(self, blocks, outputs, **kwargs):
		super().__init__(**kwargs);
		self.outputs = outputs;
		self.block_configs = [{**block, 'skip': outputs} for block in blocks];
		self.blocks = [ConvResnetBlock(**config) for config in self.block_configs];
		
	def call(self, input, constants = None):
		input_shape = tf.shape(input);
		
		if(constants is not None):
			constant_shape = tf.shape(constants);
			
			"""constants = tf.broadcast_to(
				constants,
				shape = [input_shape[0], input_shape[1], constant_shape[-1]],
				name = 'conditioning'
			);"""
			with tf.name_scope("conditioning"):
				constants = constants + tf.zeros(dtype=constants.dtype, shape = [input_shape[0], input_shape[1], constant_shape[-1]]);
		
		output = tf.zeros(shape = [input_shape[0], input_shape[1], self.outputs]);
		
		for block in self.blocks:				
			skip, input = block(input, constants);
			output = output + skip;
		
		return output;
	
	def padding(self):
		paddings = [block.padding() for block in self.blocks];
		return sum(paddings);

class Wavenet(ConvResnet):
	def __init__(self, blocks, outputs, **kwargs):
		super().__init__([{**block, 'causal' : True} for block in blocks], outputs, **kwargs);