
# coding: utf-8

# # Capstone Project: ResNet-50 for Cats.Vs.Dogs

# In[2]:


from __future__ import division
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input, Lambda, Reshape
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD, Nadam
from keras.utils.data_utils import get_file

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import random
import os
import itertools
from collections import Counter
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# In[3]:

SEED = 42
np.random.seed(SEED)


# In[4]:


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUs = get_available_gpus()


# In[5]:


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


# ## Data preprocessing
# 
# - The images in train folder are divided into a training set and a validation set.
# - The images both in training set and validation set are separately divided into two folders -- cat and dog according to their lables.
# 
# *(the two steps above were finished in  Preprocessing train dataset.ipynb)*
# 
# - The RGB color values of the images are rescaled to 0~1.
# - The size of the images are resized to 224*224.
# 

# In[6]:


image_width = 224
image_height = 224
image_size = (image_width, image_height)
BATCH_SIZE = 2000

train_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
        'mytrain',  # this is the target directory
        target_size=image_size,  # all images will be resized to 224x224
        batch_size=BATCH_SIZE+2,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
        'myvalid',  # this is the target directory
        target_size=image_size,  # all images will be resized to 224x224
        batch_size=BATCH_SIZE,
        class_mode='binary')


# In[7]:


# x, y = train_generator.next()

# plt.figure(figsize=(16, 8))
# for i, (img, label) in enumerate(zip(x, y)):
#     if i >= 18:
#         break
#     plt.subplot(3, 6, i+1)
#     if label == 1:
#         plt.title('dog')
#     else:
#         plt.title('cat')
#     plt.axis('off')
#     plt.imshow(img, interpolation="nearest")

# # Delete the dataset generated above
# del x, y


# In[8]:


class DataGenerator:
    """docstring for DataGenerator"""
    def __init__(self, X_train, y_train, num_train_pairs, num_val_pairs, 
                 num_train = 1000, batch_sz=32, verbose = False):
        self.X_train = X_train[:num_train]
        self.verbose = verbose

        pairs, y =  self.create_pairs(y_train[:num_train], num_train_pairs + num_val_pairs)
        self.tr_pairs, self.tr_y = pairs[:num_train_pairs], y[:num_train_pairs]
        self.val_pairs, self.val_y = pairs[:num_val_pairs], y[:num_val_pairs]
        
        self.samples_per_train = len(self.tr_pairs)
        self.samples_per_val = len(self.val_pairs)
        
        self.batch_sz = batch_sz
        self.cur_train_index = 0
        self.cur_val_index = 0

    def create_pairs(self, labels, num_pairs):
        train_indices = np.arange(labels.shape[0])
        pair_indices = np.array(list(itertools.combinations(train_indices, 2)))
        pair_labels = np.array([ 1.0 *(x == y) for x, y in itertools.combinations(labels, 2)])
        num_samples_generated = len(pair_labels)
        if num_pairs < num_samples_generated:
            indices = np.random.choice(num_samples_generated, num_pairs, replace = False)
            pair_indices = pair_indices[indices]
            pair_labels = pair_labels[indices]
#         pair_labels[pair_labels == 0] = -1
        if self.verbose:
            print(Counter(pair_labels))
        return (pair_indices, pair_labels)

    def next_train(self):
        while True:
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            pair_indices = self.tr_pairs[self.cur_train_index : self.cur_train_index + self.batch_sz]
            yield ([ self.X_train[pair_indices[:, 0]], self.X_train[pair_indices[:, 1]] ],
                self.tr_y[self.cur_train_index : self.cur_train_index + self.batch_sz])
    
    def next_val(self):
        while True:
            self.cur_val_index += self.batch_sz
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index=0
            pair_indices = self.val_pairs[self.cur_val_index : self.cur_val_index + self.batch_sz]
            yield ([ self.X_train[pair_indices[:, 0]], self.X_train[pair_indices[:, 1]] ],
                self.val_y[self.cur_val_index : self.cur_val_index + self.batch_sz])


# ## Build the structure of ResNet-50 for Cats.Vs.Dogs
# 
# 1. Build the structure of ResNet-50 without top layer.
# 2. Add top layer to ResNet-50.
# 3. Setup training attribute.
# 4. Compile the model.

# In[9]:


X_train, y_train = train_generator.next()
X_test, y_test = test_generator.next()


# ### 1.Build the structure of ResNet-50 without top layer. 
# Pass the train and test data throught the network and del the model from memory

# In[8]:


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
K.clear_session()
size = (image_width, image_height, 3)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=size)
flatten = Flatten()(base_model.output)
model = Model(inputs = base_model.input, outputs = flatten)


# In[9]:


# Train data
bottleneck_features_train = model.predict(X_train)
# save the output as a Numpy array
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

# Test data
bottleneck_features_test = model.predict(X_test)
# save the output as a Numpy array
np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)
del base_model, model


# 
# ### Build the Model

# In[38]:


from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

INPUT_SHAPE = 2048
reg = 1e-4
base_network = Sequential()
base_network.add(Dense(1024, input_shape=(INPUT_SHAPE,), 
kernel_regularizer = l2(reg)))
base_network.add(BatchNormalization())
base_network.add(Activation('tanh'))#LeakyReLU(alpha = 0.03))
base_network.add(Dense(512, kernel_regularizer = l2(reg)))
base_network.add(BatchNormalization())
base_network.add(Activation('tanh'))#LeakyReLU(alpha = 0.03))
base_network.add(Dense(256, kernel_regularizer = l2(reg), activation='tanh'))
base_network.summary()


# ### Siamese Net

# In[11]:


print(GPUs)


# In[39]:


from keras import layers

with tf.device(GPUs[0]):
    input_a = Input(shape=(INPUT_SHAPE,))
    processed_a = base_network(input_a)
# with tf.device(GPUs[1]):
    input_b = Input(shape=(INPUT_SHAPE,))
    processed_b = base_network(input_b)
    cos_distance = layers.Dot(axes = -1, normalize = True)([processed_a, processed_b])
    siamese_net = Model([input_a, input_b], cos_distance)


# In[40]:


siamese_net.summary()


# ### 7. Compile the model.

# In[41]:


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) +
                  y_true * K.square(K.maximum(margin - y_pred, 0)))


# 
# ## Train ResNet-50 for Cats.Vs.Dogs and Save the best model.

# In[42]:


train_data = np.load('bottleneck_features_train.npy')


# In[43]:


NUM_TRAIN_PAIRS = 100000
NUM_VAL_PAIRS = 20000
BATCH_SIZE = 64
datagen = DataGenerator(train_data, y_train, batch_sz = BATCH_SIZE, num_train_pairs = NUM_TRAIN_PAIRS,		num_val_pairs = NUM_VAL_PAIRS, verbose = True)

# In[44]:


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
              patience=3, verbose = 1, min_lr=1e-8)


# In[45]:


nadam = Nadam(lr=2e-3)
siamese_net.compile(optimizer=nadam, loss=contrastive_loss)


# In[47]:

NUM_EPOCHS = 120
STEPS_PER_EPOCH = NUM_TRAIN_PAIRS//BATCH_SIZE
VALIDATION_STEPS = NUM_VAL_PAIRS//BATCH_SIZE
history = siamese_net.fit_generator(
        datagen.next_train(),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NUM_EPOCHS,
        validation_data=datagen.next_val(),
        validation_steps=VALIDATION_STEPS,
        callbacks = [reduce_lr])


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('train_vs_val_loss.png')


# In[ ]:


import h5py
siamese_net.save_weights('saved_weights_model.h5', overwrite=True)
# del model


# In[ ]:


siamese_net.load_weights('saved_weights_model.h5')
test_data = np.load(open('bottleneck_features_test.npy'))


# In[ ]:


from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
    
def kernel(x, y):
    return siamese_net.predict([x, y])[:, 0]

def compute_kernel(X, Y):
    n1, n2 = X.shape[0], Y.shape[0]
    columns = [np.array([x] * n2) for x in X]    
    dot_products =[ kernel(col, Y) for col in columns]
    return np.vstack(dot_products)


n_samples = 2000
train_examples = train_data[0: n_samples]
train_kernel = compute_kernel(train_examples, train_examples)


# In[ ]:


clf = svm.SVC(kernel='precomputed')
clf.fit(train_kernel, y_train[:n_samples])


# In[ ]:


y_train_pred = clf.predict(train_kernel)
y_train_true = y_train[: n_samples]
print("Train accuracy {}".format(accuracy_score(y_train_true, y_train_pred)))
print(confusion_matrix(y_train_true, y_train_pred))

n = 1000
test_kernel = compute_kernel(test_data[:n], train_examples)
y_pred = clf.predict(test_kernel)
y_true = y_test[:n]

print("Test accuracy {}".format(accuracy_score(y_true, y_pred)))
print(confusion_matrix(y_true, y_pred))

