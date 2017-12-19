import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
import keras as K
import tensorflow as tf
import numpy as np
from keras.models import load_model
from multiprocessing import Process

from keras.backend.tensorflow_backend import set_session
from utils import limited_gpu_memory_session
set_session(limited_gpu_memory_session())

DATA_DIR = '/home/Drive2/rishabh/'

def print_metrics(y_pred, y_true):
    print("Accuracy {}".format(accuracy_score(y_true, y_pred)))
    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))

class SVM_model:
    
    def __init__(self, model_file, batch_sz = None):
        self.siamese_net = load_model(model_file, custom_objects={'tf':tf})
	self.siamese_net.load_weights(os.path.join(DATA_DIR, 'checkpointed_weights_omniglot.hdf5'))        
	if batch_sz is not None:
            self.batch_size = batch_sz
        else:
            self.batch_size = 32
        self.clf = None
    
    def fit(self, X_train, y_train, verbose = True):
        self.train_kernel = self.compute_kernel(X_train, X_train)
        C = [0.001,0.02,0.04,0.05, 0.06, 0.07,0.08, 0.1, 0.2, 0.5,0.6, 0.7,0.8,
             0.9, 1.0, 2.0, 5.0, 7.0, 10.0, 40.0, 100.0]
        max_acc = 0
        for slack in C: 
            clf = svm.SVC(C = slack, kernel='precomputed')
            clf.fit(self.train_kernel, y_train)
            y_train_pred = clf.predict(self.train_kernel)
            acc = accuracy_score(y_train, y_train_pred)
            if acc > max_acc:
                max_acc = acc
                best_C = slack
        print("{} Train Accuracy using slack parameter {}".format(max_acc, best_C))
        
        clf = svm.SVC(C = best_C, kernel='precomputed')
        clf.fit(self.train_kernel, y_train)
        self.clf = clf
        if verbose:
            y_train_pred = clf.predict(self.train_kernel)
            print_metrics(y_train, y_train_pred)
    
    def kernel(self, x, y):
        return self.siamese_net.predict([x, y])

    def compute_kernel(self, X, Y):
        n = Y.shape[0]
        columns = (np.array([x] * n) for x in X)    
        dot_products =[self.kernel(col, Y) for col in columns]
        return (np.hstack(dot_products)).T
    
    def predict(self, X_test):
        test_kernel = compute_kernel(X_test, self.X_train)
        return self.clf.predict(test_kernel)
    
    def predict_from_kernel(self, test_kernel):
        return self.clf.predict(test_kernel)
