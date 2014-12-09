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
import time
from sklearn.externals import joblib
from operator import itemgetter

from multiprocessing import Process, Queue, Pipe

Q = Queue()
TQ = Queue()
# to split the work to difference processors
def grouplen(sequence, chunk_size):
	a = list(zip(*[iter(sequence)] * chunk_size))
	tail = len(sequence)%chunk_size
	if tail!=0:
		print [sequence[len(sequence)-tail+t] for t in range(tail)]
		a.append([sequence[len(sequence)-tail+t] for t in range(tail)])
	return a

class ProcessWorker(Process):
	"""
	This class runs as a separate process to execute worker's commands in parallel
	Once launched, it remains running, monitoring the task queue, until "None" is sent
	"""
	def __init__(self, iSvm, samples, labels, c_value=0.5):
		Process.__init__(self)
		self.iSvm = iSvm
		self.samples = samples
		self.labels = labels
		self.c_value = c_value

	def run(self):
		clf = svm.SVC(C=self.c_value, class_weight ='auto')
		clf.fit(self.samples, self.labels)
		Q.put((self.iSvm, clf))
		print "Classifier #", self.iSvm, "Trained"
		return

class ProcessTest(Process):
	"""
	This class runs as a separate process to execute worker's commands in parallel
	Once launched, it remains running, monitoring the task queue, until "None" is sent
	"""
	def __init__(self, iSvm, model, samples, labels):
		Process.__init__(self)
		self.iSvm = iSvm
		self.samples = samples
		self.labels = labels
		self.model = model

	def run(self):
		print "Testing Classifier #", self.iSvm
		# clean the test samples
		use_test_samples = self.samples[self.labels!=0]
		use_test_attributes = self.labels[self.labels!=0]
		predicted_score = self.model.decision_function(use_test_samples)
		# predicted labels
		predicted_result = self.model.predict(use_test_samples)
		## calculate the average precision
		avg_precision = average_precision_score(use_test_attributes, predicted_score)
		## calculate the accuracy
		accuracy = accuracy_score(use_test_attributes, predicted_result)
		TQ.put([self.iSvm, avg_precision, accuracy])

class ProcessPredict(Process):
	"""
	This class runs as a separate process to execute worker's commands in parallel
	Once launched, it remains running, monitoring the task queue, until "None" is sent
	"""
	def __init__(self, iSvm, model, samples):
		Process.__init__(self)
		self.iSvm = iSvm
		self.samples = samples
		self.model = model

	def run(self):
		print "Predicting Classifier #", self.iSvm
		# clean the test samples
		# predicted labels
		predicted_result = self.model.predict(self.samples)
		TQ.put([self.iSvm, predicted_result])



def train(train_samples, train_attributes, c_values, get_interval=1):
	# start the test process, thus every item in the queue will be consumed
	t = Process(target=collect, args=(get_interval,train_attributes.shape[0],) )
	t.start()
	# my machine only has four cores
	p_list = grouplen(range(train_attributes.shape[0]), 4)
	for pp in p_list:
		## start a loop of processing
		pros = []
		for pos in pp:
			use_train_attr = train_attributes[pos, train_attributes[pos,:]!=0]
			use_train_samples = train_samples[train_attributes[pos,:]!=0]
			p = ProcessWorker(pos, use_train_samples, use_train_attr, c_values[pos])
			p.start()
			pros.append(p)
		for p in pros:
			p.join()
			print 'joined'

def collect(get_interval, csize):
	# retrieve the result from the Queue
	collected_models = []
	while len(collected_models)!=csize:
		while not Q.empty():
			item = Q.get()
			collected_models.append(item)
			print "Get,", len(collected_models)
		time.sleep(get_interval)

	# put the result in the queue
	# sort list according to the index of the model
	models = sorted(collected_models, key=itemgetter(0))
	Q.put ( [m[1] for m in models] )

# save indicates whether to save the models
# the results always get saved. savename indicates the filename to save
def test(test_samples, test_attributes, save=True, savename="model_metrics"):
	# retrieve the models from the queue
	models = Q.get()
	# split the models
	m_list = grouplen(range(len(models)), 4)
	for mm in m_list:
		pros = []
		for m in mm:
			p = ProcessTest(m, models[m], test_samples, test_attributes[m,:])
			p.start()
			pros.append(p)
		for p in pros:
			p.join()
			print 'Test joined'
	# get the results
	results = []
	while len(results)!=len(models):
		item = TQ.get()
		results.append(item)
	print '#',len(models),'results retrieved!'
	#sort the result
	results = sorted(results, key=itemgetter(0))
	precisions = [r[1] for r in results]
	accuracies = [r[2] for r in results]

	print "Mean Precision", np.mean(precisions)
	print "Mean Accuracy", np.mean(accuracies)

	# save the metric results
	f = h5py.File(savename+".hdf5",'w-')
	f.create_dataset("precisions", data=precisions)
	f.create_dataset("accuracies", data=accuracies)
	f.close()	
	if save:
		# save the models
		for m in models:
			joblib.dump(m[1],"./saved_models/svm_model_"+str(m[0])+".pkl")

# save indicates whether to save the models
# the results always get saved. savename indicates the filename to save
def predict(test_samples, save=True, savename="predicted_results"):
	# retrieve the models from the queue
	models = Q.get()
	# split the models
	m_list = grouplen(range(len(models)), 4)
	for mm in m_list:
		pros = []
		for m in mm:
			p = ProcessPredict(m, models[m], test_samples)
			p.start()
			pros.append(p)
		for p in pros:
			p.join()
			print 'Prediction joined.'
	# get the results
	results = []
	while len(results)!=len(models):
		item = TQ.get()
		results.append(item)
	print '#',len(models),'results retrieved!'
	#sort the result
	results = sorted(results, key=itemgetter(0))
	predicts = np.array([p[1] for p in results])
	print "Check predicts shape,", predicts.shape

	# save the predicted results
	if save:
		f = h5py.File(savename+".hdf5",'w-')
		f.create_dataset("predicted_labels", data=predicts)
		f.close()	



if __name__ == "__main__":
	print "Usage: being called within another script"
	# start = timeit.default_timer()
	# stop = timeit.default_timer()
	# print "Time Used,", round(stop - start, 4)