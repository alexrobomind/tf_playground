import tensorflow as tf

def swap_axes(data, ax1, ax2):
	perm = tf.range(0, tf.rank(data));
	perm = tf.tensor_scatter_nd_update(perm, ax1, ax2);
	perm = tf.tensor_scatter_nd_update(perm, ax2, ax1);
	
	return tf.transpose(data, perm = perm);

class BlockConv2D(tf.keras.Model):
	def __init__(self, channels, blocks_in, blocks_out, transpose = False, use_bias = False, activation = None, **kwargs):
		super().__init__();
				
		self.blocks_in = blocks_in;
		self.blocks_out = blocks_out;
		self.channels = channels;
		self.transpose = transpose;
		self.activation = activation;
		self.use_bias = use_bias;

		self.kwargs = kwargs;
		
		self.convlayers = None;
		self.bias = None;
		self.act_layer = None;
	
	@tf.function
	def call(self, input):
		n_blocks = self.blocks_in.shape[0];
		
		assert n_blocks is not None, "shape[0] of blocks_in must be known statically"; # No. of blocks must be known
		assert self.blocks_out.shape[1] is not None, "shape[1] of blocks_out must be known statically"; # Size of block in output must be known
		assert self.blocks_out.shape[0] == n_blocks, "shape[0] of blocks_out must match shape[0] of blocks_in"; # No. of blocks must be same for in and out
		
		inputs = tf.gather(input, self.blocks_in, axis = 3);
		inputs = tf.unstack(inputs, num = n_blocks, axis = 3);
		
		if self.convlayers is None:
			self.convlayers = [
				self.make_single_op()
				for i in range(0, self.blocks_in.shape[0])
			];
		
		assert len(self.convlayers) == n_blocks;
		
		outputs = [l(i) for l,i in zip(self.convlayers, inputs)];
		outputs = tf.stack(outputs, axis = -2);
		
		# Since there is no "scatter" operation, we need to use scatter_nd instead
		# Input dimensions: Batch x Height x Width x Channel Block x Channel in Block
		# Output dimensions: Channel Block x Channel in Block x Batch x Height x Width
		outputs = tf.transpose(outputs, perm = [3, 4, 0, 1, 2]);
		
		# Calculate shapes pre- and post-scatter
		pre_shape = tf.shape(outputs);
		post_shape = [self.channels, pre_shape[2], pre_shape[3], pre_shape[4]];
		
		blocks_out = self.blocks_out;
		blocks_out = tf.reshape(blocks_out, [tf.shape(blocks_out)[0], tf.shape(blocks_out)[1], 1]);
		#print('Blocks: ');
		#print(blocks_out);
		#print('Non-Scattered outputs:');
		#print(outputs);
		
		outputs = tf.scatter_nd(blocks_out, outputs, post_shape);
		outputs = tf.transpose(outputs, [1, 2, 3, 0]);
		#print('Scattered outputs:');
		#print(outputs);
		
		if self.use_bias:
			if self.bias is None:
				with tf.init_scope():
					bias_init = tf.zeros(dtype = tf.float32, shape = (self.channels,));
					self.bias = tf.Variable(bias_init);
			
			outputs = outputs + self.bias;
		
		if self.activation is not None:
			if self.act_layer is None:
				self.act_layer = tf.keras.layers.Activation(self.activation);
			
			outputs = self.act_layer(outputs);
		
		return outputs;
	
	def make_single_op(self):
		if self.transpose:
			return tf.keras.layers.Conv2DTranspose(
				filters = self.blocks_out.shape[1],
				use_bias = False,
				**self.kwargs
			);
		else:
			return tf.keras.layers.Conv2D(
				filters = self.blocks_out.shape[1],
				use_bias = False,
				**self.kwargs
			);

class RandomSparseConv2D(tf.keras.Model):
	def __init__(self, channels, n_blocks, blocksize_in, blocksize_out, **kwargs):
		super().__init__();
		
		self.channels = channels;
		self.n_blocks = n_blocks;
		self.bs_in = blocksize_in;
		self.bs_out = blocksize_out;
		
		self.blocks_in = None;
		self.blocks_out = None;
		
		self._encoder = None;
		self.decoder = None;
		
		self.kwargs = kwargs;
		
		self.n_channels_in = None;
	
	def build(self, input_shape):
		self.n_channels_in = input_shape[-1];
	
	@tf.function
	def call(self, input):
		if self._encoder is None:
			with tf.init_scope():
				n_channels_in = self.n_channels_in;
				
				def make_perm(n_blocks, block_size, n_channels):
					perm = tf.range(0, n_blocks * block_size) % n_channels;
					perm = tf.random.shuffle(perm);
					perm = tf.reshape(perm, [n_blocks, block_size], name = "block_permutation");
					return perm;
				
				# Initialize index permutations
				self.blocks_in =  make_perm(self.n_blocks, self.bs_in, n_channels_in);
				self.blocks_out = make_perm(self.n_blocks, self.bs_out, self.channels);
			
				self.blocks_in  = tf.Variable(self.blocks_in , trainable = False);
				self.blocks_out = tf.Variable(self.blocks_out, trainable = False);
			
			self._encoder = BlockConv2D(self.channels, self.blocks_in, self.blocks_out, transpose = False, **self.kwargs);
			self.decoder  = BlockConv2D(n_channels_in, self.blocks_out, self.blocks_in, transpose = True , **self.kwargs);
		
		return self._encoder(input);
	
	def encoder(self, input):
		return self(input);

class EncoderDecoderStack(tf.keras.Model):
	def __init__(self, layers):
		super().__init__();
		
		self.l = layers;
	
	#@tf.function
	def encoder(self, input):
		for l in self.l:
			input = l.encoder(input);
		
		return input;
	
	#@tf.function
	def decoder(self, input):
		for l in reversed(self.l):
			input = l.decoder(input);
		
		return input;
	
	def call(self, input):
		fw = self.encoder(input);
		bw = self.decoder(fw);
		
		return bw;