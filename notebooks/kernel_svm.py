import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
import keras as K
import tensorflow as tf
import numpy as np
from keras.models import load_model

import kernel_util; reload(kernel_util)
from kernel_util import SVM_model, print_metrics
from keras.preprocessing.image import ImageDataGenerator

from keras.backend.tensorflow_backend import set_session
from utils import *
set_session(limited_gpu_memory_session())

image_width, image_height = 105, 105
image_size = (image_width, image_height)
DATA_DIR = '/home/Drive2/rishabh'

TEST_FEATURES = os.path.join(DATA_DIR, 'features_test_omniglot.npy')

image_width = 105
image_height = 105
image_size = (image_width, image_height)

datagen = ImageDataGenerator(rescale=1.0/255)

test_dir = os.path.join(DATA_DIR, 'omniglot_keras/images_evaluation')
test_generator = datagen.flow_from_directory(
        test_dir,  target_size=image_size, # this is the target directory
        batch_size = 13180, color_mode="grayscale",
        class_mode='sparse')

alphabet_to_index = get_alphabet_to_index(test_generator)
X_val, y_val = test_generator.next()

unique_labels = np.unique(y_val)
matches = {}
for x in unique_labels:
    matches[x] = np.where(y_val == x)[0]

test_data = X_val#np.load(TEST_FEATURES)
#baseline_siamese_path = os.path.join(DATA_DIR, 'baseline_siamese_omniglot.h5')
baseline_siamese_path =  os.path.join(DATA_DIR, 'siamese_omniglot1.h5')
svm_model = SVM_model(baseline_siamese_path)

acc = []
for alph in alphabet_to_index:
    for _ in range(2):
        class_arr = alphabet_to_index[alph]
        sample_classes = np.random.choice(class_arr, 20, replace = False)
        drawers = np.random.choice(20, 2, replace = False)
        train_arr = [matches[x][drawers[0]] for x in sample_classes] 
        test_arr = [matches[x][drawers[1]] for x in sample_classes]
        X_small, y_small = test_data[train_arr], y_val[train_arr]
        X_test, y_test = test_data[test_arr], y_val[test_arr]
        probs = svm_model.compute_kernel(X_test, X_small)
        class_indices = np.argmax(probs, axis = 1)
        acc.append(np.sum(class_indices == np.arange(20))/20.0)
print(acc)
print(np.mean(acc))
"""
svm_model = SVM_model(MODEL_FILE)
probs = svm_model.compute_kernel(X_val[test_arr], X_val[train_arr])
print(np.argmax(probs, axis = 0))
del svm_model

svm_model = SVM_model(baseline_siamese_path)
probs = svm_model.compute_kernel(test_data[test_arr], test_data[train_arr])
print(np.argmax(probs, axis = 0))
del svm_model

svm_model = SVM_model(baseline_siamese_path)
svm_model.fit(X_small, y_small)

train_kernel = svm_model.train_kernel
train_kernel2 = np.dot(train_kernel, train_kernel.T)
svm_model.clf.fit(train_kernel2, y_small)
y_pred = svm_model.predict_from_kernel(train_kernel2)
print_metrics(y_small, y_pred)
print(y_pred)

test_kernel = svm_model.compute_kernel(X_test, X_small)
print(test_kernel.shape)

test_kernel2 = np.dot(test_kernel.T, train_kernel)
print(test_kernel2.shape)

y_test_pred = svm_model.predict_from_kernel(test_kernel2)
print_metrics(y_test, y_test_pred)
"""