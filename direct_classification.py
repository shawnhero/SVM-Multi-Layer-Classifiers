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


if __name__ == "__main__":
	start = timeit.default_timer()

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
	print "Check feature shape", features.shape


	kf = KFold(filenames.shape[0], n_folds=5)


	## set the label to predict here
	labels = politic_label_dem_gop
	
	precisions = []
	accuracies = []

	for train, test in kf:
		print "Folding..."
		train_features = features[train,:]
		train_labels = labels[train]
		test_features = features[test,:]
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

	stop = timeit.default_timer()
	print "Train Time Used,", round(stop - start, 4)
	# result
	# Mean Precision: 0.790512321089
	# Mean Accuracies 0.697872340426
	