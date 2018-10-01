import tensorflow as tf

import tensorflow.data
import tensorflow.train
import tensorflow.python_io
import tensorflow.summary

import os.path as path
import os

import scipy.io as sio
import scipy.io.wavfile

import numpy as np

import datetime

import wavenet

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
		
		texts = [open(file, "r").read().splitlines()[0] for file in text_files];
		
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
					
					print(" --- wavedata ---");
					print(wavedata);
					
					csize = chunk_size    if chunk_size   is not None else wavedata.size;
					csize = min(csize, wavedata.size);
					
					for start in range(0, wavedata.size - csize + 1, max(csize - chunk_overlap, 1)):
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
								"region"  : speaker["region"]
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

class IndexedConstant(tf.keras.Layer):
	def __init__(self, shape):
		self.shape = shape;
	
	def build(self, input_shape):
		self.data = self.add_weight("storage", self.shape, use_resource = True);
	
	def call(self, input):
		tf.gather_nd(input, self.data);

def setup_loss(data, meta_graph_def):
	with tf.name_scope("per_speaker_data"):
		data_source = tf.get_variable(
			name = "storage",
			shape = [200, 10],
			use_resource = True
		);
		
		per_speaker_data = tf.gather_nd(data["speaker"]["id"], data_source);
		
	input_map = {
		"inputs" : data["wave"]["data"],
		"constants" : per_speaker_data
	};
	
	



if __name__ == "__main__":
	corpus = VCTKCorpus(vctk_c_folder);
	
	data = corpus.dataset(chunk_size = 48000 * 3, chunk_overlap = 48000 * 1).batch(1).make_one_shot_iterator().get_next();
	
	audio = tf.summary.audio("test", data["wave"]["data"], 48000);
	
	summaries = tf.summary.merge_all();
	
	with tf.Session() as sess:
		timestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
		writer = tf.summary.FileWriter("logdir/{}".format(timestring), sess.graph);
		
		for i in range(0, 10):
			writer.add_summary(sess.run(summaries), i);
		
		writer.close();