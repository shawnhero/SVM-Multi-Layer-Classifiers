import h5py
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import misc
from skimage.feature import hog
from skimage import data, color, exposure



#Given a image, return the truncated image containing the face
def GetFaceRect(faceimage, landmark):
	return faceimage[ int(landmark[12,1]-80):int(landmark[12,1]+60),int(landmark[12,0]-50):int(landmark[12,0]+50)]

def getHog(faceimage, landmark):
	# this will generate 8100 features for each image
	image = color.rgb2gray(faceimage)
	image = GetFaceRect(image, landmark.reshape(49,2))
	fd = hog(image, orientations=6, pixels_per_cell=(8, 8),
	                cells_per_block=(3, 3), visualise=False)
	return fd

if __name__ == "__main__":
	hfile =  h5py.File('../LFW_meta.hdf5','r')
	landmarks = np.transpose(hfile['landmark'])
	filenames = np.array(hfile['img_filenames'])
	rootpath = "./../LFW_image/"

	hogs = np.empty((filenames.shape[0], 8100),dtype=float )
	print "Total faces to get hog,", filenames.shape[0]
	for k in range(filenames.shape[0]):
		face = misc.imread(rootpath+filenames[k])
		hogs[k] = getHog(face, landmarks[k])
		if k%100==0:
			print k, "Done.."
	## save the file
	f = h5py.File("merged_features.hdf5",'w-')
	f.create_dataset("features", data=np.concatenate((
		landmarks, 
		hogs), axis=1))
	f.close()






