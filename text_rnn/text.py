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

# Create dataset

with tf.name_scope('data_source'):
	filename = "iliad.txt";
	dataset = tf.data.TextLineDataset(filename).skip(500).repeat();
	print(dataset);

	# Create tokenizer to map the strings onto sequences
	vocab_size = 100;
	tokenizer = kpt.Tokenizer(
		num_words = vocab_size,
		filters = '',
		lower = False,
		split = '',
		char_level = True,
		oov_token = None
	);

	with open(filename, 'r') as file:
		text = file.read();
	tokenizer.fit_on_texts([text]);
	
	def tk_func(x):
		def ttm(y):
			#print(list(y.decode('utf-8')));
			matrix = tokenizer.texts_to_matrix(list(y.decode('utf-8')));
			#print(matrix);
			return matrix.astype(np.float32);
			
		out = tf.py_func(
			ttm,
			[x],
			(tf.float32),
			False
		);
		out.set_shape([tf.Dimension(None), vocab_size]);
		return out;
	
	matrix_dataset = dataset.map(tk_func);
	
	print(matrix_dataset);
	
	# Batch data together and create input and output datasets by dropping the last resp. the first element
	batch_size = 200;#tf.placeholder(tf.int64);
	batched_dataset = matrix_dataset.padded_batch(
		batch_size,
		(-1, vocab_size)
	);
	
	input_dataset = batched_dataset.map(
		lambda x : x[:,0:-1,:]
	);
	
	output_dataset = batched_dataset.map(
		lambda x : x[:,1:,:]
	);
	
	print(output_dataset);
	
	# Create iterators
	input_iterator = input_dataset.make_initializable_iterator();
	output_iterator = output_dataset.make_initializable_iterator();

input = input_iterator.get_next(name='input');
reference = output_iterator.get_next(name='reference');

tf.summary.histogram('Input', tf.argmax(input, axis = 2));
tf.summary.histogram('Reference', tf.argmax(reference, axis = 2));

#temp = tf.keras.layers.Embedding(
#	input_dim = vocab_size,
#	output_dim = 30
#)(input_value);

#temp = tf.keras.layers.LSTM(
#	units = 20,
#	return_sequences = True
#)(input);

# Set up layers

class Model():
	def __init__(self):
		self.rnn_cell = tf.contrib.rnn.LSTMCell(
			200,
			initializer = tf.initializers.orthogonal()#,
			#activation=tf.sigmoid
		);
		
		#self.rnn_cell_2 = tf.contrib.rnn.LSTMCell(
		#	20,
		#	initializer = tf.initializers.orthogonal()#,
		#	#activation=tf.sigmoid
		#);

		self.dense_layer_1 = tf.keras.layers.Dense(
			20,
			activation = 'sigmoid'
		)

		self.dense_layer_2 = tf.keras.layers.Dense(
			vocab_size
		);
		
	def __call__(self, input, state = None):		
		# Set up network
		temp, state = tf.nn.dynamic_rnn(
			self.rnn_cell,
			input,
			initial_state = state,
			dtype = tf.float32
		);
		
		#temp = self.dense_layer_1(temp);
		temp = self.dense_layer_2(temp);
		
		return temp, state;
	
	def zero_state(self):
		return (self.rnn_cell.zero_state(), self.rnn_cell_2.zero_state());

model = Model();
output, _ = model(input);
print(output);
print(reference);

# Reshape both output and reference to same shape
output = tf.reshape(output, (-1, vocab_size));
reference = tf.reshape(reference, (-1, vocab_size));

# Compute data filter mask
filter_mask = (tf.reduce_sum(input, axis = 2) > 0);
filter_mask = tf.reshape(filter_mask, (-1,));

output = tf.boolean_mask(
	output,
	filter_mask
);

reference = tf.boolean_mask(
	reference,
	filter_mask
);

loss = tf.losses.softmax_cross_entropy(
	reference,
	output
);# +
#0.2 *
#loss = tf.losses.mean_squared_error(
#	reference,
#	output
#);

#weights = rnn_cell.weights[0];
#print(weights.shape[0]);
#loss = loss + 0.1 * tf.norm(
#	tf.matmul(weights, tf.transpose(weights)) - tf.eye(int(weights.shape[0]))
#);

tf.summary.scalar("Loss", loss);
tf.summary.histogram("LSTM ", model.rnn_cell.weights[0]);
tf.summary.histogram("LSTM Biases", model.rnn_cell.weights[1]);
tf.summary.histogram("Dense weights", model.dense_layer_2.weights[0]);
tf.summary.histogram("Dense biases", model.dense_layer_2.weights[1]);
tf.summary.histogram("Output", tf.argmax(output, axis = 1));

# Prepare training ops
optimizer = tf.train.AdamOptimizer(0.01);
train_op = optimizer.minimize(loss);

it_init_ops = [input_iterator.initializer, output_iterator.initializer];
init_op = tf.global_variables_initializer();

# Prepare output
i_max = 1000;

print('Building evaluation loop...');
def loop_body(i, i_max, state, distribution, outputs):
	# Run network
	distribution, state = model(distribution, state);
	print(distribution);
	print(state);
	distribution = tf.reshape(distribution, (vocab_size,));
	distribution = tf.nn.softmax(distribution);
	
	#print(distribution.shape);
	#
	## Determine output
	#output = tf.argmax(distribution, 2)[0, 0];
	output = tf.distributions.Categorical(probs=distribution).sample();
	output = tf.cast(output, dtype = tf.int32);
	#
	## Update outputs tensor
	outputs = outputs + tf.sparse_to_dense(
		[[i]],
		(i_max,),
		output
	);
	
	random = tf.random_uniform(
		(),
		1,
		50,
		dtype = tf.int32
	);
	#
	## Set new feedback input
	distribution = tf.sparse_to_dense(
		[[0,0,output]],
		(1, 1, 100),
		[1.0]
	);# + 0.7 * tf.random_uniform(()) * tf.sparse_to_dense(
	#	[[0, 0, random]],
	#	(1, 1, 100),
	#	[1.0]
	#);
	
	return (i+1, i_max, state, distribution, outputs);
	
_, _, _, _, test = tf.while_loop(
	lambda i, i_max, *unused : i < i_max,
	loop_body,
	(0, 100, model.rnn_cell.zero_state(1, tf.float32), tf.zeros(shape = (1, 1, 100)), tf.zeros(dtype = tf.int32, shape = (100,))),
	name = 'Testing'
);
	
_, _, _, distribution, outputs = tf.while_loop(
	lambda i, i_max, *unused : i < i_max,
	loop_body,
	(0, 1000, model.rnn_cell.zero_state(1, tf.float32), tf.zeros(shape = (1, 1, 100)), tf.zeros(dtype = tf.int32, shape = (1000,))),
	name = 'Inspection'
);

def map_reverse(sequences):
	reverse_mapping = {v:k for k, v in tokenizer.word_index.items()};
	output_chars = "".join([
		reverse_mapping[sequences[i]]
		for i in range(0, sequences.shape[0])
		if sequences[i] in reverse_mapping
	]);
	
	return output_chars;

tf.summary.histogram("Test result", test);
tf.summary.text("Test string", tf.py_func(
	map_reverse,
	[test],
	tf.string
));

print('Done');

merge = tf.summary.merge_all();

with tf.Session() as session:
	writer = tf.summary.FileWriter(datetime.datetime.now().strftime("output/%I_%M%p on %B %d %Y"), session.graph);
	
	session.run(init_op);
	session.run(it_init_ops);
	
	saver = tf.train.Saver();
	checkpoint_format = 'checkpoints/{0}.chkp';
	i_start = 0;
	#saver.restore(session, checkpoint_format.format(i_start));

	
	for epoch in range(i_start,10001):
		_, l, m = session.run([train_op, loss, merge]);
		
		print('Loss: {0}'.format(l));
		
				
		if(epoch % 1000 == 0):
			saver.save(session, checkpoint_format.format(epoch));
		
		writer.add_summary(m, epoch);
	
	# Try some data outputting
	print('Running evaluation loop');
	outputs_eval = session.run(outputs);
	print('Done');
	
	
	print("Free-run result:");
	print("".join(outputs_chars));
	
	
	#print(session.run(distribution));
	
	# outputs = session.run(outputs);
	print(outputs);
	print(outputs_eval);
	
	writer.close();
	
	
# Run dataset
#iter = dataset.batch(batch_size).make_initializable_iterator();
#
#session.run(iter.initializer, feed_dict = {batch_size : 60});
#print(session.run(iter.get_next()));



exit();

sequence = tokenizer.texts_to_sequences([text]);

batch_length = 5;

layers = [
	tf.keras.layers.Embedding(
		input_dim  = batch_length,
		output_dim = 20
	),
	tf.keras.layers.LSTM(
		units = 20
	),
	tf.keras.layers.Dense(
		vocab_size
	)
];

network = tf.keras.Sequential(
	layers
);

network.compile(
	optimizer = 'adam',
	loss = 'binary_crossentropy'
);

input = tf.zeros(shape=(10, 10), name = 'Input');

output = network(input);

init_op = tf.global_variables_initializer();

# Run session
with tf.Session() as session:
	session.run(init_op);