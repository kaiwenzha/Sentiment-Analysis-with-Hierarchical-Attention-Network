from __future__ import print_function, division
import os
try:
	import xml.etree.cElementTree as ET
except ImportError:
	import xml.etree.ElementTree as ET
import torch
from torch.autograd import Variable
from model import HA_NET
from constants import *
from word_embedding import load_word2vec, embedding
from preprocess import preprocess_file


def tagging(txt):
	for char in txt:
		if not ((char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z') or (
				char >= '0' and char <= '9') or char in " ,.<>/?\\!@#$%^&*()-_+=`~\t;:\'\"[]{}|"):
			return CN
	return EN

def load_my_model(Tag):
	model = HA_NET(Embedding_Dim[Tag])
	saved_state = torch.load(os.path.join('trained_models', 'model_%s.dat' % Tag_Name[Tag]))
	model.load_state_dict(saved_state)
	model = model.cuda()
	return model

def load_model(args, Tag):
    model = HA_NET(Embedding_Dim[Tag])
    saved_state = torch.load(
        os.path.join(args.model_dir, 'test_{}'.format(Tag_Name[Tag]), 'model_%s.dat' % Tag_Name[Tag]))
    model.load_state_dict(saved_state)
    if args.gpu:
        model = model.cuda()
    return model

def evaluate(args, input_file_path, out_file_path):
	# print(input_file_path)
	preprocess_file(input_file_path, Tag_Dict[args.tag])
	xmltree = ET.parse(input_file_path)
	xmlroot = xmltree.getroot()
	language_model = load_word2vec(Tag_Dict[args.tag])
	model = load_model(args, Tag_Dict[args.tag])

	counter = 0
	for review in xmlroot:
		txt = review.text
		if txt[-1] == '\n':
			txt = txt[:-1]
		# tag = tagging(txt)
		# if len(txt) != 0:
		mat = embedding(language_model, txt, Tag_Dict[args.tag])
		data = Variable(torch.from_numpy(mat))
		if args.gpu:
			data = data.cuda()
		output = model.forward(data)

		if output.data.cpu().numpy()[0] < 0.5:
			review.set("polarity", "-1")
		else:
			review.set("polarity", "1")
		# else:
		# 	counter += 1
		# 	review.set("polarity", "1")
	# print('ERROR: {}'.format(counter))
	xmltree.write(out_file_path, encoding="utf-8")

if __name__ == "__main__":
	torch.set_default_tensor_type('torch.DoubleTensor')
	language_model = list(map(lambda x: load_word2vec(x), Languages))
	model = list(map(lambda x: load_my_model(x), Languages))
	print('请输入语句：\n    ', end='')
	s = input()
	while s != 'exit':
		tag = tagging(s)
		mat = embedding(language_model[tag], s, tag)
		data = Variable(torch.from_numpy(mat)).cuda()
		output = model[tag].forward(data)
		if output.data.cpu().numpy()[0] < 0.5:
			print('\033[1;31m' + '差评' + '\033[0m')
		else:
			print('\033[1;32m' + '好评' + '\033[0m')
		print('请输入语句：\n    ', end='')
		s = input()
