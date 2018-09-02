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

def convert_to_records(dir):
	glob_in = glob.glob('{}/*_in.wav'.format(dir));
	glob_inout = glob.glob('{}/*_inout.wav'.format(dir));
	
	patterns_in = [s[:-7] for s in glob_in];
	patterns_inout = [s[:-10] for s in glob_inout];
	
	for pattern in patterns_in:
		input_filename = '{0}_in.wav'.format(pattern);
		output_filename = '{0}_out.wav'.format(pattern);
		record_filename = '{0}_record.tfrecord'.format(pattern);
	
		if do_build([input_filename, output_filename], record_filename):
			print('Building {out} out of {in1} and {in2}'.format(
				out = record_filename,
				in1 = input_filename,
				in2 = output_filename
			));
			build_record_wavpair(input_filename, output_filename, record_filename);
	
	for pattern in patterns_inout:
		stereo_filename = '{0}_inout.wav'.format(pattern);
		record_filename = '{0}_record.tfrecord'.format(pattern);
		
		if do_build([stereo_filename], record_filename):
			print('Building {out} out of {inp}'.format(
				out = record_filename,
				inp = stereo_filename
			));
			build_record_stereowav(stereo_filename, record_filename);

def do_build(input_filenames, output_filename):
	# Check if all input files exist
	if(not all([path.exists(name) for name in input_filenames])):
		return False;
	
	# Check if output file exists
	if(not path.exists(output_filename)):
		return True;
	
	# Check when the newest modification for all files was
	mod_times = [os.stat(filename).st_mtime for filename in input_filenames];
	max_mod_time = max(mod_times);
	
	output_mod_time = os.stat(output_filename).st_mtime;
	
	# Return true when one of the input files is newer
	return max_mod_time >= output_mod_time;

def build_record_stereowav(stereo_filename, record_filename):
	(rate, data) = sio.wavfile.read(stereo_filename);
	
	if(data is None):
		print('Failed to read WAV data for {}, aborting'.format(stereo_filename));
		return;
	
	data = convert_wavdata_to_int32(data);
	
	if(data.shape[1] == 1):
		print('{0} has only one channel, duplicating it'.format(stereo_filename));
		data = np.repeat(data, 2, axis = 1);
	
	write_record(rate, data, record_filename);

def build_record_wavpair(input_filename, output_filename, record_filename):
	(rate1, data1) = sio.wavfile.read(input_filename);
	(rate2, data2) = sio.wavfile.read(output_filename);
	
	if(data1 is None):
		print('Failed to read WAV data for {}, aborting'.format(input_filename));
		return;
	
	if(data2 is None):
		print('Failed to read WAV data for {}, aborting'.format(output_filename));
		return;
	
	# Make sure the rates match
	if(rate1 != rate2):
		print('Files {ifile} and {ofile} have conflicting sampling rate {irate} and {orate}, skipping...'.format({
			ifile : input_filename,
			ofile : output_filename,
			irate : rate1,
			orate : rate2
		}));
		return;
	
	# Use left channel of stereo files
	if(data1.shape[1] > 1):
		data1 = data1[:,0];
		
	if(data2.shape[1] > 1):
		data2 = data2[:,0];
	
	# Convert datatype
	data1 = convert_wavdata_to_int32(data1);
	data2 = convert_wavdata_to_int32(data2);

	# Merge data into single array
	lentarget = max(len(data1), len(data2));
	data = np.zeros(shape = [lentarget, 2]);
	data[0:len(data1), 0] = data1;
	data[0:len(data2), 1] = data2;
	
	# Build record
	write_record(rate1, data, record_filename);

def write_record(rate, data, filename):
	def bytes_feature(data):
		return tf.train.Feature(
			bytes_list = tf.train.BytesList(
				value = data
			)
		);
	
	def int64_feature(data):
		return tf.train.Feature(
			int64_list = tf.train.Int64List(
				value = data
			)
		);
	
	features = {
		'rate' : int64_feature([rate]),
		'shape' : int64_feature(list(data.shape)),
		'data' : bytes_feature([data.tostring()])
	};
	
	example = tf.train.Example(
		features = tf.train.Features(
			feature = features
		)
	);
	
	writer = tf.python_io.TFRecordWriter(filename);
	writer.write(example.SerializeToString());
	writer.close();
	
# Converts a waveform from all supported WAV formats to int32
def convert_wavdata_to_int32(input):
	# 8 bit types are unsigned, so they first need to be converted to a higher width and then down-shifted
	if(input.dtype == np.uint8):
		input = input.astype(np.int16);
		
		delta = np.iinfo(np.int8).max - np.iinfo(np.uint8).max;
		input = input + delta;
	
		# Rescale data (a bit-shift won't work because that would mess with the left-most sign bit)
		input = input * 0x100;
	
	# Re-scale 16 bit data to 32 bit
	if(input.dtype == np.int16):
		input = input.astype(np.int32);
		input = input * 0x10000;
	
	if(input.dtype == np.float32):
		absmax = np.amax(np.abs(input));
		
		# Normalize files to [-1, 1]
		if(absmax > 1):
			input = input / absmax;
		
		# Scale to correct size and convert
		input = input * np.iinfo(np.uint32).max;
		input = input.astype(np.int32);
	
	return input;

def convert_to_float(from_type, to_type):
	from_max = 1 if from_type.is_floating else from_type.max;
	from_min = -1 if from_type.is_floating else from_type.min;
	
	return lambda input: (
		
	);
		
		
convert_to_records('.');