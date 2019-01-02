#coding:utf-8
from __future__ import print_function, division
import os
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from constants import *
from preprocess import *

def transfer2polar(Tag, input_file_path, out_file_path):
	xmltree = ET.parse(input_file_path)
	xmlroot = xmltree.getroot()

	for idx, review in enumerate(xmlroot):
		if xmlroot[idx].get('label') == '0':
			xmlroot[idx].set('label', '-1')
		xmlroot[idx].attrib['polarity'] = xmlroot[idx].attrib.pop('label')

	xmltree.write(out_file_path, encoding="utf-8")

def producetest(Tag, input_file_path):
	xmltree = ET.parse(input_file_path)
	xmlroot = xmltree.getroot()

	element_negative = ET.Element('reviews')
	element_positive = ET.Element('reviews')
	for review in xmlroot:
		if review.get('label') == '0':
			# sub = ET.SubElement(element_negative, 'review')
			# sub.text = review.text
			element_negative.append(review)
		else:
			# sub = ET.SubElement(element_positive, 'review')
			# sub.text = review.text
			element_positive.append(review)

	newtree_positive = ET.ElementTree(element_positive)
	newtree_negative = ET.ElementTree(element_negative)

	newtree_positive.write(os.path.join(Dataset_Dir, '{}_test'.format(Tag_Name[Tag]), 'test.positive.xml'), encoding='utf-8')
	newtree_negative.write(os.path.join(Dataset_Dir, '{}_test'.format(Tag_Name[Tag]), 'test.negative.xml'), encoding='utf-8')

def processans():
	for lan in Languages:
		preprocess_file(os.path.join(Dataset_Dir, 'ans', 'test.label.{}.xml'.format(Tag_Name[lan])), lan)
		# transfer2polar(lan, os.path.join(Dataset_Dir, 'ans', 'test.label.{}.xml'.format(Tag_Name[lan])),
		#                os.path.join(Dataset_Dir, 'ans', 'test.submit.{}.xml'.format(Tag_Name[lan])))
		producetest(lan, os.path.join(Dataset_Dir, 'ans', 'test.label.{}.xml'.format(Tag_Name[lan])))

if __name__ == '__main__':
	processans()
