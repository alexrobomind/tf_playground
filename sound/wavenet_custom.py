import tensorflow as tf

import tensorflow.keras
#import tensorflow.keras.layers
import tensorflow.contrib.rnn

import tensorflow.summary

import numpy as np

class CausalConv1D(tf.keras.layers.Conv1D):
	def __init__(
		self,
		**kwargs
	):
		super().__init__(
			padding='valid',
			strides=1,
			**{k, v for k in kwargs.items() if k not in ['padding', 'strides']}
		);
		
	def call(self, input):
		padding = self.dilation_rate[0] * (self.kernel_size[0] - 1);
		padded_input = tf.pad(input, tf.constant([[0, 0], [padding, 0], [0, 0]]));
		return super().call(padded_input);

class CausalConv1DRNNCell(tf.contrib.rnn.RNNCell):
	def __init__(
		self,
		units,
		kernel_size,
		dilation=1,
		trainable=True,
		name=None,
		dtype=None,
		**kwargs
	):
		super().__init__(trainable=trainable,name=name,dtype=dtype,**kwargs);
		
		self.kernel_size = kernel_size;
		self.units = units;
		self.dilation = dilation;
		
		self.dt = (kernel_size - 1) * dilation;
		#self.state_size = self.units * (self.dt + 1);
		#self.state_size =  # Use a (self.dt+1)-rank tuple of tensors as state (makes the shuffling and scattering very efficient)
		
	@property
	def state_size(self):
		return (self.units,) * (self.dt + 1);
	
	def build(self, input_shape):
		if(input_shape[1].value is None):
			raise ValueError("Expecting a shape of rank 2");
		
		self.kernel_shape = (input_shape[1], self.kernel_size, self.units);
		
		# Define variable self.kernel [shape ]
		self.kernel = self.add_weight(
			name = 'kernel',
			shape = self.kernel_shape
		);
	
	def call(self, input, state):
		batch_size = input.shape[0];
		
		if(state is None):
			state = (tf.zeros(shape = (batch_size, self.units), name='initial_state_slice'),) * (self.dt + 1);
		
		# Reshape state to the right structure (now unneccessary)
		# state = tf.reshape(state, shape = (batch_size, self.units));
			
		# The convolution formula for a single unit is Out[t] = Sum (t', t' <= K_size) K[i, t'] * In[t-self.dt+d*t', i]
		#  meaning, that the lowest kernel matrix element is the one that goes furthest back in time.
		# The state tensor at time t holds at [:, j, :] the partially constructed convolution result of Out[t + j].
		
		# Insert new row at end, drop the first row
		state = state[1:];
		state = state + (tf.zeros(shape = (batch_size, self.units), name = 'zero_state_slice'),);
		#state = tf.concat([
		#	state[:, 1:, :],
		#	tf.zeros(shape = (batch_size, 1, self.units))
		#], axis = 1);
		
		# Calculate the update tensor (by using left-hand multiplication
		# The resulting tensor has shape (batch_size, self.kernel_size, self.units)
		update_tensor = tf.tensordot(
			input,
			self.kernel,
			axes = [[1], [0]]
		);
		
		# The slice [:, 0, :] of update_tensor corresponds to t = t' - self.dt (the lowest kernel element) when looking from the output at t' into the past.
		# Since we are looking forward into the future from In, we are providing data in that slice that need to live for dt times. Therefore
		# they need to be scattered to [:, self.dt, :]. From that we can derive the broadcasting formula
		#
		# [:, j, :] -> [:, self.dt - self.dilation * j, :]
		src_indices = np.full(fill_value = -1, shape = [self.dt + 1], dtype = np.int64);
		for j in range(0, self.kernel_size):
			src_indices[self.dt - self.dilation * j] = j;
		
		new_state = ();
		for j in range(0, self.dt + 1):
			if src_indices[j] != -1:
				update_subtensor = update_tensor[:, src_indices[j], :];
				update_subtensor = tf.reshape(update_subtensor, shape = (batch_size, self.units));
				
				new_state = new_state + (tf.add(state[j], update_subtensor, name='updated_state'),);
			else:
				new_state = new_state + (state[j],);
		
		#indices = numpy.zeros(shape = (batch_size, self.kernel_size, 2));
		#for i in range(0, batch_size):
		#	for j in range(0, self.kernel_size):
		#		indices[i, j, 0] = i;
		#		indices[i, j, 1] = self.dt - self.dilation * j;
		#		
		#update_tensor_scattered = tf.scatter_nd(
		#	indices = indices,
		#	updates = update_tensor,
		#	shape = state.shape,
		#	name = 'Updates'
		#);
		#
		#state = state + update_tensor_scattered;
		
		output = state[0];
		
		return (output, new_state);
	
class WaveNetBlock(tf.keras.layers.Layer):
	def __init__(self, output_units, skip_units, kernel_size, **kvargs):
		super().init(__kwargs__);
		
	def build(self, input_shapes):
		self.conv_blocks = (
			CausalConv1D(output_units, kernel_size, activation = 'sigmoid', name='conv_sigmoid'),
			CausalConv1D(output_units, kernel_size, activation = 'tanh', name = 'conv_tanh')
		;
		
		if(input_shape[1] != output_units)
			self.residual_projector = tf.keras.layers.Conv1D(output_units, 1, name = 'project_residual');
		else
			self.residual_projector = None;
		
		if(skip_units != output_units)
			self.skip_projector = tf.keras.layers.Conv1D(skip_units, 1, name = 'project_skip');
		else
			self.skip_projector = None;
			
	def call(self, input):
		conv_out = conv_blocks[0](input) * conv_blocks[1](input);
		
		if(self.residual_projector)
			residual = self.residual_projector(input);
		else
			residual = input;
		
		if(self.skip_projector)
			skip = self.skip_projector(conv_out);
		else
			skip = conv_out;
		
		return (conv_out + residual, skip);

class WaveNet(tf.keras.layers.Layer):
# Returns a base-mu logarithm encoder
def mu_enconde(num_channels):
	mu = tf.to_float(num_channels) - 1;
	
	return lambda input: (
		tf.sign(input) *
		tf.log(1 + mu * tf.abs(audio)) / tf.log(1 + mu)
	);

# Returns a base-mu logarithmic decoder
def mu_decode(num_channels):
	mu = tf.to_float(num_channels) - 1;
	
	return lambda input: (
		tf.sign(input) * (
			#tf.exp(tf.abs(input) * tf.log(1 + mu) - 1
			(1+mu)**tf.abs(input) - 1
		) / mu
	);
		
#	def __init__(self, )