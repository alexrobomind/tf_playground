import tensorflow as tf

import tensorflow.data
import tensorflow.train
import tensorflow.python_io
import tensorflow.summary

import os.path as path
import os

import scipy.io as sio
import scipy.io.wavfile

import glob

import numpy as np

import datetime

import wavenet

import tkinter as tk
import tkinter.filedialog

def model(input, condition = None):
	blocks = [{
		'outputs': 512,
		'hidden': [{'filters': 256, 'kernel_size': 2, 'dilation_rate': 2**i}]
	} for i in range(0, 10)];
	
	nets = [wavenet.Wavenet(blocks, 256) for i in range(0, 1)];
	
	for net in nets:
		input = net(input, condition);
	
	padding = sum([net.padding() for net in nets]);
	
	for i in range(0, 2):
		input = tf.keras.layers.Conv1D(
			filters = 256,
			kernel_size = 1,
			activation = 'relu'
		)(input);
		
	n_distributions = 10;
	
	return tf.keras.layers.Conv1D(
		filters = 3 * n_distributions,
		kernel_size = 1
	)(input), padding;

if __name__ == "__main__":
	constant_size = int(input("Specify a number of constant slots for the model: "));
	
	tk_root = tk.Tk();
	tk_root.withdraw();
	filename = tk.filedialog.asksaveasfilename(initialdir=".", title="Select filenames for metagraph", filetypes = [("All files", "*.*")]);
	
	print("Building model graph ...");

	inputs = tf.placeholder(shape = [None, None, 1], name = 'inputs', dtype=tf.float32);
	constants = tf.placeholder(shape = [constant_size], name = 'constants', dtype=tf.float32) if constant_size > 0 else None;
	outputs = model(inputs, constants);
	
	outputs = tf.identity(outputs, name = 'outputs');
	
	initializer = tf.global_variables_initializer();
	
	print("Starting TensorFlow session ...");
	with tf.Session() as sess:
		print("Initializing model ...");
		sess.run(initializer);
		
		print("Saving model to {} ...".format(filename))
		tf.train.Saver().save(sess, filename);