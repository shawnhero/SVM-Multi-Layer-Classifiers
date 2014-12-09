import multi_svm
import h5py
import numpy as np
import timeit
from scipy import misc
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from add_hog_feature import GetFaceRect, getHog


def first_train():
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
	multi_svm.train(features[is_train==1], attributes[:,is_train==1], c_values,  get_interval=5)


def first_predict(num_models, prediction_save_name):
	hfile =  h5py.File('../Politic_meta.hdf5','r')
	politic_label_win_loss = np.array(hfile['politic_label_win_loss'])
	politic_label_dem_gop = np.array(hfile['politic_label_dem_gop'])
	landmarks = np.transpose(hfile['landmark_poli'])
	filenames = np.array(hfile['img_filenames_poli'])
	
	rootpath = "./../Politic_image/"
	
	## calculate the hog features
	hogs = np.empty((filenames.shape[0], 8100),dtype=float )
	print "Total faces to get hog,", filenames.shape[0]
	for k in range(filenames.shape[0]):
		face = misc.imread(rootpath+filenames[k])
		hogs[k] = getHog(face, landmarks[k])
		if k%50==0:
			print k, "Done.."
	## concatenate the features
	features = np.concatenate((landmarks, hogs), axis=1)
	print "Politic_image features generated! Check shape:", features.shape
	multi_svm.predict(features, num_models, save=True, savename=prediction_save_name)


def second_train_and_test(attributes_load_name):
	#do the cross validations here

	#first load the attributes file
	hfile =  h5py.File(attributes_load_name+'.hdf5','r')
	attibutes = np.transpose( hfile['predicted_labels'] )
	print "Check feature shape", attibutes.shape
	kf = KFold(attibutes.shape[0], n_folds=5)

	#then load the win/loss data
	wfile =  h5py.File('../Politic_meta.hdf5','r')
	politic_label_win_loss = np.array(wfile['politic_label_win_loss'])
	politic_label_dem_gop = np.array(wfile['politic_label_dem_gop'])

	labels = politic_label_dem_gop
	# set the labels to predict here
	precisions = []
	accuracies = []
	for train, test in kf:
		print "Folding..."
		train_features = attibutes[train,:]
		train_labels = labels[train]
		test_features = attibutes[test,:]
		test_labels = labels[test]

		clf = svm.SVC(C=1000.0)
		clf.fit(train_features, train_labels)
		predicted_score = clf.decision_function(test_features)
		# predicted labels
		predicted_result = clf.predict(test_features)
		## calculate the average precision
		avg_precision = average_precision_score(test_labels, predicted_score)
		precisions.append(avg_precision)
		## calculate the accuracy
		accuracy = accuracy_score(test_labels, predicted_result)
		accuracies.append(accuracy)
		print confusion_matrix(test_labels,predicted_result)

	print 'Mean Precision:', np.mean(precisions)
	print 'Mean Accuracies', np.mean(accuracies)
	##4 attributes used
	# Mean Precision: 0.679121065159
	# Mean Accuracies 0.617021276596


if __name__ == "__main__":
	start = timeit.default_timer()
	
	first_train()
	stop = timeit.default_timer()
	print "Train Time Used,", round(stop - start, 4)

	first_predict(68, prediction_save_name="politics_predicts")
	stop2 = timeit.default_timer()
	print "Predict Time Used,", round(stop2 - stop, 4)

	second_train_and_test(attributes_load_name="politics_predicts")
	stop3 = timeit.default_timer()
	print "Second Train Time Used,", round(stop3 - stop2, 4)
	