import tensorflow as tf

import tensorflow.data
import tensorflow.train
import tensorflow.python_io
import tensorflow.summary

import wavenet_model

import os.path as path
import os

import scipy.io as sio
import scipy.io.wavfile

import numpy as np

import datetime

import wavenet
import multi_logistics

import tkinter as tk
import tkinter.filedialog

import librosa

import csv

vctk_c_folder = "D:\Downloads\VCTK_Corpus\VCTK-Corpus\VCTK-Corpus";

class VCTKCorpus:
	def __init__(self, folder):
		self.folder = folder;
		
	def speakers(self):
		with open(self.folder + "/speaker-info.txt", "r") as csvfile:
			iterator = iter(csvfile);
			keys = [s.lower() for s in next(iterator).split()];
			
			dictionaries = [
				{key : value for (key, value) in zip(keys, line.split())}
				for line in iterator
			];
			
			return {dict["id"] : dict for dict in dictionaries};
	
	def speaker_data(self, id):
		wave_dir = "{folder}/wav48/p{id}".format(id = id, folder = self.folder);
		text_dir = "{folder}/txt/p{id}".format(id = id, folder = self.folder);
		
		prefixes = ['.'.join(name.split('.')[:-1]) for name in os.listdir(text_dir)];
		
		text_files = ["{dir}/{prefix}.txt".format(dir = text_dir, prefix = prefix) for prefix in prefixes];
		wave_files = ["{dir}/{prefix}.wav".format(dir = wave_dir, prefix = prefix) for prefix in prefixes];
		
		def readfile(filename):
			with open(filename, "r") as file:
				return file.read();
				
		texts = [readfile(file).splitlines()[0] for file in text_files];
		
		return [{"text" : text, "wave" : wave} for text, wave in zip(texts, wave_files)];
	
	# A dataset that contains the whole VCTK corpus w/ speaker data in order
	def dataset(self, sampling_rate = 48000, chunk_size = None, chunk_overlap = 0):
		assert chunk_overlap >= 0;
		assert chunk_size is None or chunk_size > 0;
		assert sampling_rate > 0;
		
		def generator():
			speakers = self.speakers();
			speaker_ids = [int(i) for i in speakers];
			
			id_begin = min(speaker_ids);
			id_end = max(speaker_ids) + 1;
			
			for id, speaker in speakers.items():
				speaker_data = self.speaker_data(id);
				
				for entry in speaker_data:
					wavedata, _ = librosa.core.load(entry["wave"], sr = sampling_rate, mono = True);
					
					csize = chunk_size    if chunk_size   is not None else wavedata.size;
					csize = min(csize, wavedata.size);
					
					for start in list(range(0, wavedata.size - csize + 1, max(csize - chunk_overlap, 1))):
						yield {
							"text"        : entry["text"],
							"wave" : {
								"data" : wavedata[start : start + csize],
								"file" : entry["wave"],
								"offset" : start
							},
							"speaker" : {
								"id"      : int(speaker["id"]),
								"age"     : int(speaker["age"]),
								"gender"  : speaker["gender"],
								"accents" : speaker["accents"],
								"region"  : speaker["region"] if "region" in speaker else ""
							},
							"dataset" : {
								"speaker_id_begin" : id_begin,
								"speaker_id_end"   : id_end
							}
						};
		
		types = {
			"text" : tf.string,
			"wave" : {
				"data"   : tf.float32,
				"file"   : tf.string,
				"offset" : tf.int32
			},
			"speaker" : {
				"id"      : tf.int32,
				"age"     : tf.int32,
				"gender"  : tf.string,
				"accents" : tf.string,
				"region"  : tf.string
			},
			"dataset" : {
				"speaker_id_begin" : tf.int32,
				"speaker_id_end"   : tf.int32
			}
		};
		
		shapes = {
			"text" : tf.TensorShape([]),
			"wave" : {
				"data"   : tf.TensorShape([None]),
				"file"   : tf.TensorShape([]),
				"offset" : tf.TensorShape([])
			},
			"speaker" : {
				"id"      : tf.TensorShape([]),
				"age"     : tf.TensorShape([]),
				"gender"  : tf.TensorShape([]),
				"accents" : tf.TensorShape([]),
				"region"  : tf.TensorShape([])
			},
			"dataset" : {
				"speaker_id_begin" : tf.TensorShape([]),
				"speaker_id_end"   : tf.TensorShape([])
			}
		};
		
		return tf.data.Dataset.from_generator(
			generator,
			types,
			shapes
		);

def setup(data, meta_graph_or_file):
	with tf.variable_scope("per_speaker_data"):
		data_source = tf.get_variable(
			name = "storage",
			shape = [200, 100],
			use_resource = True
		);
		
		indices = tf.reshape(data["speaker"]["id"] - data["dataset"]["speaker_id_begin"], shape = [-1, 1, 1]);
		per_speaker_data = tf.gather_nd(data_source, indices);
	
	with tf.name_scope("inputs"):
		input = data["wave"]["data"][:, 0:-1];
		input_shape = tf.shape(input);
		input = tf.reshape(input, [input_shape[0], input_shape[1], 1]);
	
	with tf.name_scope("model"):
		# Ensure that the "inputs" and "constants" tensors exist in the model, either if it is loaded or created
		input = tf.identity(input, name = "inputs");
		per_speaker_data = tf.identity(per_speaker_data, name = "constants");
		
		if meta_graph_or_file != "":
			# Load model, if a filename is given
			input_map = {
				"inputs" : input,
				"constants" : per_speaker_data
			};
			
			saver = tf.train.import_meta_graph(
				meta_graph_or_file,
				input_map = input_map
			);
			
			graph = tf.get_default_graph();
			outputs = graph.get_tensor_by_name(graph.get_name_scope() + "/outputs:0");
			padding = 0;
		else:
			# Create model
			outputs, padding = wavenet_model.model(input, per_speaker_data);
			outputs = tf.identity(outputs, name = 'outputs');
			saver = None;
		
		with tf.name_scope("strip_padding"):
			outputs = outputs[:, padding:, ...];
		
		with tf.name_scope("reshape_for_pdf"):
			output_size = tf.shape(outputs);
			outputs = tf.reshape(outputs, shape = [output_size[0], output_size[1], -1, 3]);
		
		outputs = tf.check_numerics(outputs, "Invalid outputs");
	
	with tf.name_scope("reference"):
		reference = data["wave"]["data"][:, 1 + padding:];
	
	"""with tf.name_scope("output_postprocessing"):
		# Cut off the gradient backflow where the distribution scale is 0 (and thus the distribution is degenerate)
		dynamic_shape = tf.shape(outputs);
		outputs_flat = tf.reshape(outputs, shape = [-1, 3]);
		outputs_flat = tf.where(
			tf.less_equal(outputs_flat[:, 2], 0),
			tf.ones_like(outputs_flat),
			outputs_flat
		);
		outputs = tf.reshape(outputs_flat, shape = dynamic_shape);
		outputs = tf.Print(outputs, [outputs], summarize = 300);"""
	
	with tf.name_scope("loss"):
		with tf.name_scope("distribution"):
			distribution = multi_logistics.distribution(outputs);
		
		with tf.name_scope("sample"):
			sample = distribution.sample();
		
		#reference = tf.Print(reference, [reference], summarize=10);
		loss = -distribution.log_prob(reference);
		#loss = tf.Print(loss, [loss], summarize=10);
		# Remove NaN and Inf value from loss function
		"""loss = tf.where(
			tf.is_finite(loss),
			loss,
			tf.zeros_like(loss)			
		);"""
		loss = tf.reduce_sum(loss);
		loss = tf.check_numerics(loss, "Invalid loss");
	
	return saver, outputs, [data_source], sample, loss;
	
	

if __name__ == "__main__":
		# Request for file paths
		tk_root = tk.Tk();
		tk_root.withdraw();
		
		meta_filename = tk.filedialog.askopenfilename(initialdir=".", title="Select filenames for metagraph", filetypes = [("TF Meta Graph", "*.meta")]);
		
		if meta_filename != "":
			model_checkpoint_filename = tk.filedialog.askopenfilename(initialdir=".", title="Select filenames for metagraph", filetypes = [("TF Variable Storage", "*.index")]);
			model_checkpoint_filename = checkpoint_filename[:-6];
		
		# Open VCTK corpus
		corpus = VCTKCorpus(vctk_c_folder);
		dataset = corpus.dataset(chunk_size = 48000 * 1, chunk_overlap = int(48000 * 0.5));
		dataset = dataset.repeat();
		dataset = dataset.shuffle(buffer_size = 30000);
		dataset = dataset.padded_batch(1, dataset.output_shapes);
		dataset = dataset.prefetch(5);
		data = dataset.make_one_shot_iterator().get_next();
		
		# Setup training
		submodel_saver, output, setup_vars, sample, loss = setup(data, meta_filename);
		optimizer = tf.train.AdamOptimizer(learning_rate = 1e-6);
		step = optimizer.minimize(loss);
		
		# Setup checkpoints
		global_saver = tf.train.Saver();
		checkpoint_prefix = tk.filedialog.asksaveasfilename(initialdir=".", title="Select prefix for checkpoints");
		
		checkpoint_filename = tk.filedialog.askopenfilename(initialdir=".", title="Select filename for checkpoint", filetypes = [("TF Variable Storage", "*.index")]);
		if checkpoint_filename != "":
			checkpoint_filename = checkpoint_filename[:-6];
		
		#if submodel_saver:
		#	init = tf.initialize_variables(optimizer.variables() + setup_vars);
		#else:
		#	init = tf.initializers.global_variables();
		init = tf.initializers.global_variables();
		
		# Generate summaries
		audio_in  = tf.summary.audio("input", data["wave"]["data"], 48000);
		audio_out = tf.summary.audio("sample", sample, 48000, 1);
		tf.summary.scalar("loss", loss);
		summaries = tf.summary.merge_all();
		
		# Configure session
		config = tf.ConfigProto(log_device_placement=False);
		config.gpu_options.per_process_gpu_memory_fraction = 0.6;
		#config.gpu_options.allow_growth = True;
		
		with tf.Session(config = config) as sess:
			timestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
			writer = tf.summary.FileWriter("logdir/{}".format(timestring), sess.graph);
			
			sess.run(init);
			
			if submodel_saver:
				submodel_saver.restore(sess, model_checkpoint_filename);
			if checkpoint_filename != "":
				global_saver.restore(sess, checkpoint_filename);
			
			def save(iteration):
				global_saver.save(sess, checkpoint_prefix, global_step = iteration);
		
			j = 0;
			
			try:
				for i in range(0, 100001):
					j = i;
					
					do_trace = False;
					if do_trace:
						run_md = tf.RunMetadata();
						run_opt = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE);
						run_opt.trace_level = tf.RunOptions.FULL_TRACE;
						
						vloss, _, vsummaries = sess.run([loss, step, summaries], run_metadata = run_md, options = run_opt);
					
						writer.add_run_metadata(run_md, "step {}".format(i));
					else:
						vloss, _, vsummaries = sess.run([loss, step, summaries]);
						
					writer.add_summary(vsummaries, i);
					
					if i % 500 == 0:
						save(i);
						
					print("{step} - {loss}".format(step = i, loss = vloss));
			except KeyboardInterrupt:
				print("--- Keyboard interrupt detected on iteration {} ---".format(j));
				print("Saving...");
				save(j);
				print("Done, proceeding with termination");
						
			writer.close();