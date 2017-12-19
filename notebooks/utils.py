from __future__ import division
import numpy as np
import tensorflow as tf
from collections import Counter
import itertools
from multiprocessing import Process, cpu_count
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator
import cPickle as pickle
import os
from joblib import Parallel, delayed

N_CPU = cpu_count()
np.random.seed(42)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def limited_gpu_memory_session(frac=0.5):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = frac
    config.gpu_options.allow_growth = True 
    return tf.Session(config=config)

def generate_class_pairs(i, matches, alphabet_to_index, num_pairs):
        np.random.seed(i)
        alphabets =  alphabet_to_index.keys()
        sampled_alphs = np.random.choice(alphabets, num_pairs, replace = True)
        classes = [np.random.choice(alphabet_to_index[alph], 2) for alph in sampled_alphs]
        drawers = [np.random.choice(20, 2, replace = False) for _ in sampled_alphs]
        x1 = [(matches[cls[0]][drawer[0]], matches[cls[0]][drawer[1]]) for drawer, cls in zip(drawers, classes)]
        x2 = [(matches[cls[0]][drawer[0]], matches[cls[1]][drawer[1]]) for drawer, cls in zip(drawers, classes)]
        return np.concatenate((x1, x2))
                        
def get_alphabet_to_index(generator): 
    alphabet_to_index = {}
    for key, val in generator.class_indices.iteritems():
        alphabet = "_".join(key.split('_')[:-1])
        if alphabet in alphabet_to_index:
            alphabet_to_index[alphabet].append(val)
        else:
             alphabet_to_index[alphabet] = [val]
    return alphabet_to_index

class RandomTransformer(object):
        
    def __init__(self):
        self.data_transformer = None
        self.s = None
        
    def create_data_transformer(self, rotation_range=10, width_shift_range=0.1, 
                              height_shift_range=0.1, shear_range=0.1, zoom_range=0.1):
        self.data_transformer = ImageDataGenerator(rotation_range = rotation_range, 
                            width_shift_range = width_shift_range, 
                            height_shift_range = height_shift_range, 
                            shear_range = shear_range, 
                            zoom_range = zoom_range )
    
    def random_transform(self, X):
        return np.array([self.data_transformer.random_transform(x) for x in X])
    

def shuffle(X, y):
    s = np.arange(y.shape[0])
    np.random.shuffle(s)
    return (X[s], y[s])

class DataGenerator(RandomTransformer):
    """docstring for DataGenerator"""
    def __init__(self, X_train, y_train, num_train_pairs, num_val_pairs, 
                 X_val = None, y_val = None, num_train = None, batch_sz=32, 
                 train_alphabet_to_index=None, val_alphabet_to_index= None, 
                 verbose = False, generate_val_data = True):
        
        super(DataGenerator, self).__init__()
        self.verbose = verbose

        if generate_val_data and (X_val is None):
            assert(num_train is not None)
            self.X_train, self.X_val = X_train[:num_train], X_train[num_train:]
            y_val, y_train = y_train[num_train:], y_train[:num_train]
        else:
            self.X_train, self.X_val = X_train, X_val

        print("Generating Training Pairs..")
        self.tr_pairs, self.tr_y = self.create_pairs(y_train, num_train_pairs,
                                                    train_alphabet_to_index) 
        self.samples_per_train = len(self.tr_pairs)
        self.train_alphabet_to_index = train_alphabet_to_index
        
        
        if generate_val_data:
            print("Generating Validation Pairs..")
            self.val_pairs, self.val_y = self.create_pairs(y_val, num_val_pairs, 
                                                           val_alphabet_to_index)
            self.samples_per_val = len(self.val_pairs)
            self.val_alphabet_to_index = val_alphabet_to_index
            
        self.batch_sz = batch_sz
    
    @staticmethod
    def create_prob_dist(labels):
        label_dict = Counter(labels)
        for k, v in label_dict.items():
            label_dict[k] = 1.0/(2*v)
        return [label_dict[i] for i in labels]
        
        
    def create_pairs(self, labels, num_pairs, alphabet_to_index):
        ulabels = np.unique(labels)
        matches, no_matches = dict(), dict()
        for x in ulabels:
            matches[x] = np.where(labels == x)[0]
        
        num_alphabets = len(alphabet_to_index.keys())
        num_pairs_per_alph = num_pairs // (N_CPU * 2)
        
        with Parallel(n_jobs = N_CPU) as parallel:
            indices = parallel(delayed(generate_class_pairs)(i, matches, alphabet_to_index, 
                            num_pairs_per_alph) for i in range(N_CPU))
            pair_indices = np.array([val for sublist in indices for val in sublist])
        
        plabels = np.concatenate((np.ones(num_pairs_per_alph), np.zeros(num_pairs_per_alph)))
        pair_labels = np.tile(plabels, N_CPU)

        return shuffle(pair_indices, pair_labels)

    def next_train(self):
        self.cur_train_index = 0
        while True:
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
                self.tr_pairs, self.tr_y = shuffle(self.tr_pairs, self.tr_y)
            pair_indices = self.tr_pairs[self.cur_train_index : self.cur_train_index + self.batch_sz]            
            yield ([ self.random_transform(self.X_train[pair_indices[:, 0]]), 
                   self.random_transform(self.X_train[pair_indices[:, 1]]) ],
                self.tr_y[self.cur_train_index : self.cur_train_index + self.batch_sz])
            self.cur_train_index += self.batch_sz
    
    def next_val(self):
        self.cur_val_index = 0
        while True:
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            pair_indices = self.val_pairs[self.cur_val_index : self.cur_val_index + self.batch_sz]
            yield ([ self.X_val[pair_indices[:, 0]], 
                     self.X_val[pair_indices[:, 1]] ],
                self.val_y[self.cur_val_index : self.cur_val_index + self.batch_sz])
            self.cur_val_index += self.batch_sz

def generate_triplet_indices(candidates, matches, no_matches):
    triplets = []
    for x in candidates:
        while len(matches[x]) < 2:
            x = candidates[np.random.randint(0, len(labels))]
        idx_a, idx_p = np.random.choice(matches[x], 2, replace=False)
        idx_n = np.random.choice(no_matches[x], 1)[0]
        triplets.append([idx_a, idx_p, idx_n])
    return triplets
    

class TripletGenerator(RandomTransformer):
    
    def __init__(self, X_train, y_train, X_val, y_val, train_alphabet_to_index,
                 test_alphabet_to_index, num_train_triplets, 
                 num_val_triplets = 20000, random_transform = False,
                 batch_sz=32):
        
        self.X_val, self.X_train = X_val, X_train
        self.num_train_triplets = num_train_triplets
        self.use_random_transform = random_transform
                
        self.train_matches, self.train_nomatches = self.generate_dicts(y_train, train_alphabet_to_index)
        self.val_matches, self.val_nomatches = self.generate_dicts(y_val, test_alphabet_to_index)

        self.val_triplets = self.generate_triplets(y_val, num_val_triplets, train = False)        
        self.triplets = self.generate_triplets(y_train, num_train_triplets)
        
        self.batch_sz = batch_sz
        
    def generate_dicts(self, labels, alphabet_to_index):
        matches, no_matches = {}, {}
        for _, chars in alphabet_to_index.iteritems():
            for x in chars:
                matches[x] = np.where(labels != x)[0]
            all_matches = np.hstack([matches[x] for x in chars])
            for x in chars:
                no_matches[x] = np.setdiff1d(all_matches, matches[x])
        return matches, no_matches

    def generate_triplets(self, labels, num_triplets, train = True):
        if train:
            matches, no_matches = self.train_matches, self.train_nomatches
        else:
            matches, no_matches = self.val_matches, self.val_nomatches
        
        candidate_indices = np.random.randint(0, len(labels), size=num_triplets)
        candidates = labels[candidate_indices]
        split_len = (num_triplets // N_CPU) + 1
        
        with Parallel(n_jobs = N_CPU) as parallel:
            partitions = [(i*split_len, (i+1)*split_len) for i in range(N_CPU)] 
            triplet_indices = parallel(delayed(generate_triplet_indices)
                        (candidates[partitions[i][0]: partitions[i][1]], matches,                           no_matches) for i in range(N_CPU))
        triplets  = [val for sublist in triplet_indices for val in sublist]
        return np.array(triplets)
    
    def next_train(self):
        self.cur_train_index = 0
        while True:
            if self.cur_train_index >= len(self.triplets):
                self.cur_train_index = 0
            triplet_indices = self.triplets[self.cur_train_index : 
                        self.cur_train_index + self.batch_sz]
            X_batch = [self.X_train[triplet_indices[:, 0]], 
                self.X_train[triplet_indices[:, 1]], 
                self.X_train[triplet_indices[:, 2]] ]
            if self.use_random_transform:
                X_batch = [self.random_transform(x) for x in X_batch]
            yield (X_batch, np.ones(len(triplet_indices)) )
            self.cur_train_index += self.batch_sz
            
    def next_val(self):
        self.cur_val_index = 0 
        while True:
            if self.cur_val_index >= len(self.val_triplets):
                self.cur_val_index = 0
            triplet_indices = self.val_triplets[self.cur_val_index  :self.cur_val_index + self.batch_sz]
            X_batch = [self.X_val[triplet_indices[:, 0]], 
                     self.X_val[triplet_indices[:, 1]], 
                     self.X_val[triplet_indices[:, 2]]]
            if self.use_random_transform:
                X_batch = [self.random_transform(x) for x in X_batch]
            yield (X_batch, np.ones(len(triplet_indices)))
            self.cur_val_index += self.batch_sz
    
    def update_batch_size(batch_size):
        self.batch_size = batch_size
        

def plot_figure(history, name):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.set_figheight(8)
    f.set_figwidth(14)

    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train_loss' 'val'], loc='upper left')

    f.savefig('{}.png'.format(name))

        
"""
class SiameseGenerator:
    
    def __init__(self, X, y, X_train, y_train val_generator, batch_sz=32):
        
        self.train_generator = train_generator
        self.get_val_data(val_generator)
        self.get_train_data()
        self.batch_sz = batch_sz
    
    def get_val_data(self, val_generator, frac = 0.1):
        self.X_val, y_val = val_generator.next()
        self.val_pairs, self.val_labels = self.generate_pairs(y_val, frac)
        self.samples_per_val = len(self.val_pairs)
    
    def get_train_data(self, frac = 0.1):
        self.X_train, y_train = self.train_generator.next()
        self.pairs, self.labels = self.generate_pairs(y_train, frac)
        self.samples_per_train = len(self.pairs)

    def generate_pairs(self, labels, frac):
        train_indices = np.arange(labels.shape[0])
        pair_indices = np.array(list(itertools.combinations(train_indices, 2)))
        pair_labels = np.fromiter(((l1 == l2) for l1, l2 in 
                                   itertools.combinations(labels, 2)), np.float)
        num_samples_generated = len(pair_labels)
        num_pairs = int(frac * num_samples_generated)
        indices = np.random.choice(num_samples_generated, num_pairs, replace = False)
        pair_indices = pair_indices[indices]
        pair_labels = pair_labels[indices]        
        return (pair_indices, pair_labels)
    
    def next_train(self):
        self.cur_train_index = 0
        while True:
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
                self.get_train_data()
            pair_indices = self.pairs[self.cur_train_index : self.cur_train_index + self.batch_sz]
            pair_labels = self.labels[self.cur_train_index : self.cur_train_index + self.batch_sz]
            self.cur_train_index += self.batch_sz
            yield ([ self.X_train[pair_indices[:, 0]], self.X_train[pair_indices[:, 1]]], pair_labels)

    def next_val(self):
        self.cur_val_index = 0 
        while True:
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            pair_indices = self.val_pairs[self.cur_val_index : self.cur_val_index + self.batch_sz]
            pair_labels = self.val_labels[self.cur_val_index : self.cur_val_index + self.batch_sz]
            self.cur_val_index += self.batch_sz
            yield ([ self.X_val[pair_indices[:, 0]], self.X_val[pair_indices[:, 1]]], pair_labels)

#     def create_pairs(self, labels, num_pairs, save_to_file = None):
#         train_indices = xrange(labels.shape[0])
#         pair_indices = np.fromiter(itertools.chain(*itertools.combinations(train_indices, 2)),
#                                    dtype=np.int).reshape(-1, 2)
#         pair_labels = np.fromiter((1.0 *(x == y) for x, y in 
#                                    itertools.combinations(labels, 2)), dtype=np.float)
#         num_samples_generated = len(pair_labels)
#         if num_pairs < num_samples_generated:
#             prob_arr = DataGenerator.create_prob_dist(pair_labels)
#             indices = np.random.choice(num_samples_generated, num_pairs, replace = False, p = prob_arr)
#             pair_indices, pair_labels = pair_indices[indices], pair_labels[indices]
        
#         if self.verbose:
#             print(Counter(pair_labels))
#         if save_to_file is not None:
#             pickle.dump((pair_indices, pair_labels), open(save_to_file, "wb"))
#         return (pair_indices, pair_labels)

"""