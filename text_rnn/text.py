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

#tf.enable_eager_execution();

# This class provides (via its data attribute and the initializer op) access to a sequence dataset taken right
# out of a datafile.
class DataSource():
	def __init__(
		self,
		filename,
		name = 'data_source',
		input_name = 'input',
		output_Name = 'reference',
		vocab_size = 100,
		batch_size = 200,
		skip = 500
	):
		with tf.name_scope(name):
			# Create tokenizer to map the strings onto sequences
			self.tokenizer = kpt.Tokenizer(
				num_words = vocab_size,
				filters = '',
				lower = False,
				split = '',
				char_level = True,
				oov_token = None
			);
			
			# Fit tokenizer on text
			with open(filename, 'r') as file:
				text = file.read();
			self.tokenizer.fit_on_texts([text]);
			
			# Store reverse mapping
			self.reverse_mapping = {v : k for k, v in self.tokenizer.word_index.items()};
			
			# Create primary dataset
			# This dataset contains a rank-0 tensor (scalar) per line
			# Each such tensor is of type 'string'
			filename = "iliad.txt";
			dataset = tf.data.TextLineDataset(filename).skip(skip).repeat();
			
			# Convert string dataset into a binary category dataset
			# Each ()-tensor (containing one string) is created into a (l, v) tensor where
			#  - l is the length of the string
			#  - v is the size of the vocabulary
			# where output[i, j] == 1 if tokenizer.word_index[input[i]] == tokenizer.tokens[j] else 0
			# This is done by wrapping the tokenizer's "texts_to_matrix"-method in a "py_func" tensorflow op
			# (which maps a single entry) and calling dataset.map with a function that returns a single such op
			# (and sets the shape of the output tensor)
			def tokenize_op(x):
				def tokenize(y):
					# Split the string into single-character lists (using the 'list' constructor)
					# and call the texts_to_matrix method.
					matrix = self.tokenizer.texts_to_matrix(list(y.decode('utf-8')));
					
					# Convert to float type
					return matrix.astype(np.float32);
				
				# Wrap a call to the tokenize function with a float32 result
				out = tf.py_func(
					tokenize,     # Target function
					[x],          # Arguments
					(tf.float32), # Return type (must be specified in advance because the function is called on demand)
					False         # Whether this operation is stateful
				);
				
				# Add some shape information on the output for the tensorflow shape inference engine
				out.set_shape([tf.Dimension(None), vocab_size]);
				return out;
			
			matrix_dataset = dataset.map(tokenize_op);
						
			# Batch data together by padding all sequences to the longest one
			# In this case the datasets have shapes (l_i, vocab_size) and the operation produces
			# the shapes (batch_size, max(l_i), vocab_size)
			batched_dataset = matrix_dataset.padded_batch(
				batch_size,
				(-1, vocab_size)
			);
			
			# Create input and output data by making datasets that take all but the last resp. the first element
			input_dataset = batched_dataset.map(
				lambda x : x[:,0:-1,:]
			);
			
			output_dataset = batched_dataset.map(
				lambda x : x[:,1:,:]
			);
						
			# Create iterators
			input_iterator = input_dataset.make_initializable_iterator();
			output_iterator = output_dataset.make_initializable_iterator();
			
			# Create action that just runs both initializer ops
			init_actions = [input_iterator.initializer, output_iterator.initializer];
			with(tf.control_dependencies(init_actions)):
				self.initializer = tf.no_op('Initializer');
			
			# See below
			input_raw = input_iterator.get_next();
			output_raw = output_iterator.get_next();
				
		# Create action that retrieves data 
		# Do this outside of the name scope so that people see the nodes
		# nicely next to the data-source node coming out of it
		# The nodes are each control-dependency linked to each other's sources so that the
		# data-sets always step forward together, even if only one is used.
		with tf.control_dependencies([input_raw, output_raw]):
			self.data = (
				tf.identity(input_raw,  name = name + '_input'),
				tf.identity(output_raw, name = name + '_ouutput')
			);
	
	def map_reverse(self, sequence):
		return "".join([
			self.reverse_mapping[e]
			for e in sequence.tolist()
			if e in self.reverse_mapping
		]);

# This class holds the trainable model.
class Model(tf.keras.layers.Layer):
	def __init__(self, vocab_size, lstm_units = 200, **kwargs):
		super().__init__(**kwargs);
		
		self.lstm_units = lstm_units;
		self.dense_units = vocab_size;
	
	# Called by the __call__ method to make up the model's internal structure on the first call
	def build(self, input_shapes):
		super().build(input_shapes);
		
		# Set up an RNNCell that holds the LSTM
		self.rnn_cell = tf.contrib.rnn.LSTMCell(
			self.lstm_units,
			initializer = tf.initializers.orthogonal()
		);

		# Create a dense layer that post-processes the LSTM output
		self.dense_layer = tf.keras.layers.Dense(
			self.dense_units
		);
	
	# Called by Layer::__call__ to evaluate the model on an input.
	# Applies the model to a (batch_size, sequence_length, vocabulary_size) tensor and
	# produces a tensor of the same shape. 
	def call(self, input, state = None):		
		# Set up an RNN based on the RNN cell and call it
		temp, state = tf.nn.dynamic_rnn(
			self.rnn_cell,         # RNNCell object that supplies the recursive structure
			input,                 # Input sequence data
			initial_state = state, # 
			dtype = tf.float32
		);
		
		# Call the dense post-processing layer
		output = self.dense_layer(temp);
		
		return output, state;
	
	# Return a (shaped) initial state that this model can run on
	def zero_state(self, batch_size):
		return self.rnn_cell.zero_state(batch_size, dtype = tf.float32);
	
	# Add summaries for this model
	def summary(self):
		tf.summary.histogram(self.name + "_lstm_weights", self.rnn_cell.weights[0]);
		tf.summary.histogram(self.name + "_lstm_biases", self.rnn_cell.weights[1]);
		tf.summary.histogram(self.name + "_dense_weights", self.dense_layer.weights[0]);
		tf.summary.histogram(self.name + "_dense_biases", self.dense_layer.weights[1]);

# Computes the loss-(error-)function between a model output and the given reference
def compute_loss(output, reference, name = 'Loss'):
	with tf.name_scope(name):
		# Assume vocabulary is the last dimension
		vocab_size = reference.shape[-1];
		
		# Reshape both output and reference to same shape
		output = tf.reshape(output, (-1, vocab_size));
		reference = tf.reshape(reference, (-1, vocab_size));
	
		# Compute data filter mask
		# All data where the vocabulary axis sums to 0 or less are filtered out.
		filter_mask = (tf.reduce_sum(input, axis = 2) > 0);
		filter_mask = tf.reshape(filter_mask, (-1,));

		# Apply filter mask
		output = tf.boolean_mask(
			output,
			filter_mask
		);

		reference = tf.boolean_mask(
			reference,
			filter_mask
		);

		# Compute the loss function
		loss = tf.losses.softmax_cross_entropy(
			reference,
			output
		);
	
	return loss;

def free_run(model, runs, vocab_size, name = 'recurrent_run', sampling_scale = 1):
	# Loop body for the inner evaluation loops
	# Arguments:
	#  - Current iteration
	#  - Max no. of iterations
	#  - Input state
	#  - Distribution of characters for sampling
	#  - Output tensor holding all characters in (sequence_length) int32 tensor
	def loop_body(i, state, distribution, outputs):
		# Reshape distribution to have a batch_size and sequence_length dimension
		distribution = tf.reshape(distribution, shape = (1, 1, -1));
		
		# Run network on input and state
		distribution, state = model(distribution, state);
		
		# Eliminate the first axis and apply softmax to get a probability distribution from output
		# Multiply the distribution by a scale before taking the exponentials, to bias it more towards
		# high-confidence samples.
		distribution = tf.reshape(distribution, (vocab_size,));
		distribution = distribution * sampling_scale;
		distribution = tf.nn.softmax(distribution);
		
		# Sample the distribution and cast the output to int
		output = tf.distributions.Categorical(probs=distribution).sample();
		output = tf.cast(output, dtype = tf.int32);
		
		# Set the i-th element of the output tensor to the sequence value
		# Note: This is a copy operation unless the execution engine optimizes it
		outputs = outputs + tf.sparse_to_dense(
			[[i]],
			(runs,),
			output
		);
		
		# Feed back a vector holding a 1 at the selected character, 0 everywhere else
		distribution = tf.sparse_to_dense(
			[[output]],    # Index of the non-zero elements
			(vocab_size,), # Shape of the output tensor (one batch, single character sequence, vocabulary)
			[1.0]          # Value of the non-zero elements
		);
		
		# Return next loop input (or final result)
		return (i+1, state, distribution, outputs);

	# Set up the loop in its own block
	with tf.name_scope(name):
		# See below
		loop_header = lambda i, *unused : i < tf.identity(runs, name = 'max_runs');
		loop_arguments = (
			0,                                                   # Run counter
			model.zero_state(1),                                 # State for the model
			tf.zeros(shape = (vocab_size,), dtype = tf.float32), # First input
			tf.zeros(dtype = tf.int32, shape = (runs,))          # Buffer to hold the output data
		);
		
		# Run a while loop using the above defined header, body and initial state
		# While loop recursively applies loop_body to its own output. The input for the first iteration
		# is given as an argument. Before every iteration it calls loop_header. If that evaluates to true
		# then it continues, otherwise it returns the current input.
		#
		# Note: while_loop is an optimized implementation. The python functions are each called only once.
		# The dataflow dependencies are then extracted and wired together into a while loop in the tensorflow
		# graph itself.
		_, _, _, output = tf.while_loop(
			loop_header,
			loop_body,
			loop_arguments,
			name = 'loop'
		);
	
	return output;

# Plumb the model, datasource and loss function together
vocab_size = 100;

print('Creating datasource...');
datasource = DataSource('iliad.txt', vocab_size = vocab_size);
(input, reference) = datasource.data;

print('Creating model...');
model = Model(vocab_size);
output, _ = model(input);

print('Creating training environment...');
loss = compute_loss(output, reference);

# Prepare training ops
optimizer = tf.train.AdamOptimizer(0.01);
train_op = optimizer.minimize(loss);

# For inspection: Create an operator that applies the model and converts the data back into a string
# This also returns a tensorflow array so it only sets up the graph computation whicha re evaluated
# lazily.
def free_run_string(runs, name, sampling_scale = 1):
	data = free_run(
		model,
		runs,
		vocab_size,
		name = name,
		sampling_scale = sampling_scale
	);
	
	return tf.py_func(
		lambda x : datasource.map_reverse(x),
		[data],
		tf.string,
		name = 'map_to_string'
	);

# Prepare two runs: One with length 100 (for during training) and one with length 10000 (for after training, somewhat slow)
print('Creating inspection environment...');
test_string_1 = free_run_string(100, 'test_1');
test_string_2 = free_run_string(100, 'test_2', sampling_scale = 2);
test_string_3 = free_run_string(100, 'test_3', sampling_scale = 3);
test_distribution = free_run(model, 100, vocab_size, 'test_hist');
final_print = free_run_string(10000, 'final_print');

# Add some things we want to see in tensorboard
tf.summary.scalar("loss", loss);
tf.summary.histogram('input', tf.argmax(input, axis = 2));
tf.summary.histogram('reference', tf.argmax(reference, axis = 2));
tf.summary.histogram("output", tf.argmax(output, axis = 1));
tf.summary.histogram("test_distribution", test_distribution);
tf.summary.text("test_string_1", test_string_1);
tf.summary.text("test_string_2", test_string_2);
tf.summary.text("test_string_3", test_string_3);
model.summary();

# An operation to print out all summaries
summarize = tf.summary.merge_all();

# This is required to make sure all variables have a valid value
init_op = tf.global_variables_initializer();

# Start a tensorflow session and execute the default graph we just set up
with tf.Session() as session:
	# This class saves all data to a subdirectory of "output" with the current datetime
	writer = tf.summary.FileWriter(
		datetime.datetime.now().strftime("output/%I_%M%p on %B %d %Y"),
		session.graph
	);
	
	# Run the initialization operations
	session.run(init_op);
	session.run(datasource.initializer);
	
	# Create a saver to save the current state in checkpoints (containing the weights) which can later be restored
	saver = tf.train.Saver();
	checkpoint_format = 'checkpoints/{0}.chkp';
	
	# Optional: Use the following line to restore a previously saved checkpoint in training
	#saver.restore(session, checkpoint_format.format(i_start));
	
	# Do the training runs (note: the increment by one in the # of runs is so the last checkpoint is written)
	i_start = 0;
	
	print('Starting training');
	for epoch in range(i_start, 2001):
		# Run the optimizer, compute the loss function and write out all summary data
		_, l, m = session.run([train_op, loss, summarize]);
		
		# Print out the loss value
		print('Loss: {0}'.format(l));
		
		# Save a checkpoint every 500 steps
		if(epoch % 500 == 0):
			saver.save(session, checkpoint_format.format(epoch));
		
		writer.add_summary(m, epoch);
	
	# Print a longer model-generated text to the console
	print(session.run(final_print));
	
	# Close the writer to flush its buffer
	writer.close();