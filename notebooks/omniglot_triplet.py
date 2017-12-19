
# coding: utf-8

# In[3]:


from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  


# In[4]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[5]:


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, Subtract, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

import cv2
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# In[6]:


from keras.backend.tensorflow_backend import set_session
from utils import limited_gpu_memory_session
set_session(limited_gpu_memory_session(0.9))


# In[7]:


DATA_DIR = '/home/Drive2/rishabh/'
TRAIN_FEATURES = os.path.join(DATA_DIR, 'features_train_omniglot.npy')
TEST_FEATURES = os.path.join(DATA_DIR, 'features_test_omniglot.npy')
INIT_WEIGHTS = os.path.join(DATA_DIR, 'init_weights_omniglot_triplet.hdf5')
CHECKPOINTED_WEIGHTS = os.path.join(DATA_DIR, 'checkpointed_weights_omniglot_triplet.hdf5')


# In[8]:


from keras.preprocessing.image import ImageDataGenerator
image_width = 105
image_height = 105
image_size = (image_width, image_height)

datagen = ImageDataGenerator(rescale=1.0/255)

train_dir = os.path.join(DATA_DIR, 'omniglot_keras/images_background') # python/

train_generator = datagen.flow_from_directory(
        train_dir,  target_size=image_size,
        batch_size = 19280,
        class_mode='sparse', color_mode="grayscale",
        shuffle=True)

test_dir = os.path.join(DATA_DIR, 'omniglot_keras/images_evaluation')

test_generator = datagen.flow_from_directory(
        test_dir,  target_size=image_size, # this is the target directory
        batch_size = 13180, color_mode="grayscale",
        class_mode='sparse')


# In[9]:


# import utils; reload(utils)
from utils import get_alphabet_to_index
test_alphabet_to_index = get_alphabet_to_index(test_generator)
train_alphabet_to_index = get_alphabet_to_index(train_generator)


# In[10]:


X_train, y_train = train_generator.next()
X_val, y_val = test_generator.next()


# In[11]:


def plot_data(X):
    counter = 0
    for img in X:
        plt.subplot(4, 5, counter+1)
        counter += 1
        plt.axis('off')
        plt.imshow(img[:,:,0], cmap = 'gray', interpolation="nearest")


# In[12]:


plot_data(X_val[:20])


# # Build the convolutional neural network

# In[ ]:


# from keras.regularizers import l2
# from keras.models import load_model

# MODEL_FILE = os.path.join(DATA_DIR, 'siamese_omniglot.h5')
# siamese_net = load_model(MODEL_FILE)
# # for layer in siamese_net.layers:
# #     layer.trainable = False
# siamese_net.summary()


# In[ ]:


# convnet = siamese_net.layers[2] #load_model(convnet_file)
# convnet.summary()


# In[ ]:


# from utils import get_available_gpus

# def create_train_test_features(model, X_train, X_val):
#     dev = get_available_gpus()
    
#     # Train data
#     print("Saving Train Features..")
#     with tf.device(dev[0]):
#         bottleneck_features_train = model.predict(X_train)
#     # save the output as a Numpy array
#     np.save(open(TRAIN_FEATURES, 'w'), bottleneck_features_train)
    
#     # Test data
#     print("Saving Test Features..")
#     with tf.device(dev[-1]):
#         bottleneck_features_test = model.predict(X_val)
#     # save the output as a Numpy array
#     np.save(open(TEST_FEATURES, 'w'), bottleneck_features_test)

# if os.path.exists(TEST_FEATURES):
#     create_train_test_features(convnet, X_train, X_val)


# In[ ]:


# # train_data = conv
# my_model = Model(inputs = convnet.inputs, outputs = convnet.get_layer('flatten_1').output)
# my_model.summary()


# In[ ]:


# baseline_siamese_path = os.path.join(DATA_DIR, 'baseline_siamese_omniglot.h5')
# if os.path.exists(baseline_siamese_path):
#     INPUT_SHAPE = 4096
#     inputs = [Input(shape=(INPUT_SHAPE,)) for _ in range(2)]
#     diff = Subtract()(inputs)
#     both = Lambda( lambda x: K.abs(x), lambda x : x)(diff)
#     prediction = siamese_net.get_layer('output')(both)
#     baseline_siamese = Model(inputs = inputs, outputs=prediction)
#     baseline_siamese.summary()
#     baseline_siamese.save(baseline_siamese_path)
# else:
#     baseline_siamese = load_model(baseline_siamese_path)


# In[ ]:


# w1 = baseline_siamese.get_weights() 
# w2 = siamese_net.layers[-1].get_weights()
# for a, b in zip(w1, w1):
#     assert (a == b).all()


# In[ ]:


# train_data = np.load(TRAIN_FEATURES)
# test_data = np.load(TEST_FEATURES)


# In[ ]:


# siamese_net.predict([X_val[:2], X_val[1:3]])


# In[ ]:


# baseline_siamese.predict([test_data[:2], test_data[1:3]])


# In[ ]:


# from keras.layers.advanced_activations import LeakyReLU
# from keras.regularizers import l2, l1
# from keras.initializers import RandomNormal

# INPUT_SHAPE = 4096
# W_init = RandomNormal(mean=0, stddev=1e-2) #'glorot_uniform'
# b_init = RandomNormal(mean= 0.5, stddev=1e-2)
# def dense_relu_bn_dropout(x, size, dropout, alpha = 0.1, reg = 0):
#     x = Dense(size, kernel_regularizer = l2(reg), kernel_initializer=W_init, bias_initializer=b_init)(x)
# #     x = BatchNormalization()(x)
#     x = Activation('selu')(x)
#     x = Dropout(dropout)(x)
#     return x

# def create_network(reg, dropout, alpha = 0.1):
#     inputs = Input(shape=(INPUT_SHAPE,))
#     x = dense_relu_bn_dropout(inputs, 4096, dropout, reg)
#     x = dense_relu_bn_dropout(x, 4096, dropout, reg)
#     x = dense_relu_bn_dropout(x, 2048, dropout, reg)
#     base_network = Model(inputs=inputs, outputs = x)
#     print(base_network.summary())
#     return base_network


# In[ ]:


# base_network = create_network(1e-3, 0.5)


# In[ ]:


# def l2_norm(x):
#     return K.sqrt(K.sum(K.square(x)))


# In[ ]:


# input_pair = [Input((INPUT_SHAPE,)) for _ in range(2)]
# outputs_base = [base_network(inp) for inp in input_pair]
# diff = Subtract()(outputs_base)
# euclidean_dist = Lambda(l2_norm, output_shape = lambda x : (x[0], 1))(diff)
# model = Model(inputs = input_pair, outputs = euclidean_dist)
# model.summary()


# In[27]:


from keras.regularizers import l2
from keras.initializers import RandomNormal

W_init = RandomNormal(mean=0, stddev=1e-2) #'glorot_uniform'
b_init = RandomNormal(mean= 0.5, stddev=1e-2)
W_dense_init = RandomNormal(mean=0, stddev = 2e-1)

input_shape = (105, 105, 1)
reg = 1e-2
#build convnet to use in each siamese 'leg'
convnet = Sequential(name="convnet")
convnet.add(Conv2D(64, (10,10), activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(reg)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128, (7,7), activation='relu', kernel_regularizer=l2(reg),
                   kernel_initializer=W_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init, bias_initializer=b_init, 
                   kernel_regularizer=l2(reg)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init, bias_initializer=b_init, 
                   kernel_regularizer=l2(reg)))
convnet.add(Flatten())
convnet.add(Dense(4096,activation="sigmoid", kernel_regularizer=l2(reg),kernel_initializer=W_dense_init, 
                  bias_initializer=b_init, name = "embedding"))
print(convnet.summary())


# In[28]:


left_input = Input(input_shape, name="input_1")
encoded_l = convnet(left_input)
# with tf.device('/gpu:1'):
right_input = Input(input_shape, name="input_2")
encoded_r  = convnet(right_input)

# merge two encoded inputs with a distance metric
diff = Subtract(name="diff")([encoded_l,encoded_r])
both = Lambda(lambda x : K.abs(x), output_shape = lambda x: x, name="abs")(diff)
prediction = Dense(1, activation='sigmoid', kernel_initializer=W_dense_init, 
                   bias_initializer = b_init, name="output")(both)

siamese_net = Model(inputs=[left_input, right_input],outputs=prediction, name="siamese_net")
siamese_net.summary()


# In[29]:


siamese_net.load_weights(os.path.join(DATA_DIR, 'checkpointed_weights_omniglot.hdf5'))


# #### Define the  triplet loss

# In[30]:


MARGIN = 0.2
def triplet_loss(y_true, y_pred): # 
    return K.mean(K.maximum(0.0, y_pred + MARGIN) - y_true * 0, axis = -1)


# #### Define the  triplet network

# In[31]:


## Define the triplet network
from keras.layers import Average
model = siamese_net
input_triples = [Input(model.input_shape[0][1:]) for _ in range(3)]
pos_output = model(input_triples[:-1])
neg_output0 = model(input_triples[1:]) 
neg_output1 = model([input_triples[0], input_triples[2]])
neg_output = Average()([neg_output0, neg_output1])
diff = Subtract()([neg_output, pos_output])
triplet_net = Model(inputs = input_triples, outputs = diff)
triplet_net.summary()
triplet_net.save_weights(INIT_WEIGHTS)


# #### Create the data generator to load batches of data

# In[17]:


import utils; reload(utils)
from utils import TripletGenerator

NUM_TRAIN_TRIPLETS = 300000
NUM_VAL_TRIPLETS = 10000
BATCH_SIZE = 200
datagen = TripletGenerator(X_train, y_train, X_val, y_val, num_val_triplets=NUM_VAL_TRIPLETS, 
                           batch_sz=BATCH_SIZE, num_train_triplets=NUM_TRAIN_TRIPLETS, 
                           train_alphabet_to_index = train_alphabet_to_index,
                           test_alphabet_to_index = test_alphabet_to_index, random_transform=True)


# In[18]:


datagen.create_data_transformer()


# In[32]:


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
              patience=3, verbose = 1, min_lr=1e-8)
early_stopping = EarlyStopping(monitor='oneshot_acc',
                              min_delta=1e-4,
                              patience=25,
                              verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath=CHECKPOINTED_WEIGHTS, verbose=1, save_best_only=True, monitor='oneshot_acc')


# In[33]:


STEPS_PER_EPOCH = 1 + (NUM_TRAIN_TRIPLETS//BATCH_SIZE)
VALIDATION_STEPS = (NUM_VAL_TRIPLETS//BATCH_SIZE) + 1


# In[34]:


import custom_callbacks; reload(custom_callbacks)
from custom_callbacks import LossHistory
loss_history = LossHistory(X_val, y_val, test_alphabet_to_index)


# In[35]:


from keras.optimizers import Adam
adam = Adam(1e-3)
triplet_net.compile(loss=triplet_loss, optimizer=adam)
triplet_net.load_weights(INIT_WEIGHTS)


# In[36]:


# triplet_net.load_weights(CHECKPOINTED_WEIGHTS)
history = triplet_net.fit_generator(
        datagen.next_train(),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=500,
        validation_data=datagen.next_val(),
        validation_steps=VALIDATION_STEPS,
        callbacks = [reduce_lr, loss_history, checkpointer, early_stopping])


# In[ ]:


triplet_net.load_weights(CHECKPOINTED_WEIGHTS)


# In[ ]:


# triplet_net.load_weights(CHECKPOINTED_WEIGHTS)
history = triplet_net.evaluate_generator(
        datagen.next_train(),
        steps=STEPS_PER_EPOCH)
print(history)

