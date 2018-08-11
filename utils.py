import os
import sys
import argparse
import datetime
import collections

import numpy as np
import tensorflow as tf


data_path="/home/a123/Documents/rcnn-lstm/simple-examples/data"

save_path='./save'

load_file = "train-checkpoint-69"

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,default=data_path, help='The path of the data for training an testing')
parser.add_argument('--load_file', type=str,default=load_file, help='The path of checkpoint file of model variable saved during training')
args = parser.parse_args()

Py3 = sys.version_info[0]==3

def read_words(filename):
	with tf.gfile.GFile(filename,"r") as f:
		if Py3:
			return f.read().replace("\n", "<eos>").split()
		else:
			return f.read().decode("utf-8").replace("\n","<eos>").split()

def build_vocab(filename):
	data = read_words(filename)

	counter = collections.Counter(data)
	count_paris = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words,_= list(zip(*count_paris))
	word_to_id = dict(zip(words,range(len(words))))
	return word_to_id

def file_to_word_ids(filename, word_to_id):
	data=read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id]

def load_data(data_path):
	train_path = os.path.join(data_path, "ptb.train.txt")
	valid_path = os.path.join(data_path, "ptb.valid.txt")
	test_path = os.path.join(data_path, "ptb.test.txt")

	word_to_id = build_vocab(train_path)

	train_data = file_to_word_ids(train_path, word_to_id)
	valid_data = file_to_word_ids(valid_path, word_to_id)
	test_data = file_to_word_ids(test_path, word_to_id)
	
	vocab_size = len(word_to_id)

	id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

	print(train_data[:10])
	print("~~~~~~~~~~~~~~")
	print(word_to_id)
	print("~~~~~~~~~~~~")
	print(vocab_size)
	print("~~~~~~~~~~~~~")
	print("".join([id_to_word[x] for x in train_data[:10]]))
	print("~~~~~~~~~~~~~~~~~~")
	return train_data, valid_data, test_data, vocab_size, id_to_word

if __name__ == "__main__":
	load_data()




