## this is for problem 3a

import multi_svm
import h5py
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import misc

if __name__ == "__main__":
	hfile =  h5py.File('../LFW_meta.hdf5','r')
	landmarks = np.transpose(hfile['landmark'])
	is_train = np.array(hfile['is_train'])
	attributes = np.array(hfile['attribute_annotation'])
	attr_names = np.array(hfile['attribute_fields'])
	filenames = np.array(hfile['img_filenames'])
	rootpath = "./../LFW_image/"

	attr_mask = np.ones(73, dtype=bool)
	#baby
	attr_mask[4] = False
	#child
	attr_mask[5] = False
	#sunglasses
	attr_mask[15] = False
	#color photo
	attr_mask[53] = False
	#indians
	attr_mask[57] = False
	attributes = attributes[attr_mask,:]
	attr_names = attr_names[attr_mask]

	#read the c value 
	cfile = h5py.File('c_values.hdf5','r')
	c_values = np.array(cfile['c'])

	multi_svm.train(landmarks[is_train==1], attributes[0:8,is_train==1], c_values,  get_interval=0.1)
	multi_svm.test(landmarks[is_train==0], attributes[:, is_train==0],save=False, savename="3a_landmark_to_attributes_another")
	#multi_svm.predict(landmarks[is_train==0], save=True, savename="predicts")