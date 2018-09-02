import tensorflow as tf

import tensorflow.keras
import tensorflow.keras.layers

import tensorflow.summary

import tensorflow.keras.preprocessing.text as kpt

import wavenet

test = wavenet.CausalConv1D(5, 5);

test_cell = wavenet.CausalConv1DCell(1, 3);

with tf.Session() as session:
	#input = tf.zeros([1, 10, 1]);
	#output = test(input);
	
	input = tf.placeholder(shape = [10, 3], dtype=tf.float32);
	
	#test_cell.build(input.shape);
	state = None;
	for i in range(0, 4):
		with tf.name_scope('iteration'+str(i)):
			output, state = test_cell(input, state);
			output_copy = tf.identity(output, name='output'); # For debugging
	
	writer = tf.summary.FileWriter("output", session.graph);
	writer.close();

