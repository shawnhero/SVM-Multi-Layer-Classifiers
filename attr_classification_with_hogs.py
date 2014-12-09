## this is for problem 3b
import multi_svm
import h5py
import timeit
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import misc

if __name__ == "__main__":
	start = timeit.default_timer()
	
	hfile =  h5py.File('../LFW_meta.hdf5','r')
	is_train = np.array(hfile['is_train'])
	attributes = np.array(hfile['attribute_annotation'])
	attr_names = np.array(hfile['attribute_fields'])
	filenames = np.array(hfile['img_filenames'])
	rootpath = "./../LFW_image/"
	# read the features
	f = h5py.File("merged_features.hdf5",'r')
	features = np.array(f['features'])

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

	multi_svm.train(features[is_train==1], attributes[:,is_train==1], c_values,  get_interval=3)

	stop = timeit.default_timer()
	print "Train Time Used,", round(stop - start, 4)

	multi_svm.test(features[is_train==0], attributes[:, is_train==0],save=False, savename="3b_landmark_with_hogs_to_attributes")

	stop2 = timeit.default_timer()
	print "Train Time Used,", round(stop2 - stop, 4)


	