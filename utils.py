import numpy as np
import tensorflow as tf
from collections import Counter
import itertools
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def limited_gpu_memory_session(frac=0.5):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = frac
    config.gpu_options.allow_growth = True 
    return tf.Session(config=config)

class DataGenerator:
    """docstring for DataGenerator"""
    def __init__(self, X_train, y_train, num_train_pairs, num_val_pairs, 
                 num_train = 1000, batch_sz=32, verbose = False, using_siamese = False):
        
        self.verbose = verbose
        self.using_siamese = using_siamese
        self.X_train = X_train[:num_train]
        self.X_val = X_train[num_train:]
    
        self.tr_pairs, self.tr_y = self.create_pairs(y_train[:num_train], num_train_pairs)
        self.val_pairs, self.val_y = self.create_pairs(y_train[num_train:], num_val_pairs)
        
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
        if self.using_siamese:
            pair_labels[pair_labels == 0] = -1
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
            yield ([ self.X_val[pair_indices[:, 0]], self.X_val[pair_indices[:, 1]] ],
                self.val_y[self.cur_val_index : self.cur_val_index + self.batch_sz])

class TripletGenerator:
    
    def __init__(self, train_generator, val_generator, num_val_triplets, batch_sz=32, num_train_triplets=100):
        
        self.train_generator = train_generator
        self.X_val, y_val = val_generator.next()
        self.val_triplets = self.generate_triplets(y_val, num_val_triplets)        
        self.samples_per_val = len(self.val_triplets)
        
        self.batch_sz = batch_sz
        self.cur_train_index = 0
        self.cur_val_index = 0 
       
        self.num_train_triplets = max(num_train_triplets, batch_sz)        
        self.X_train, y_train = self.train_generator.next()
        self.triplets = self.generate_triplets(y_train, self.num_train_triplets)
        self.cur_train_index = len(self.triplets)
    
    def generate_triplets(self, labels, num_triplets):
        triplets = []
        ulabels = np.unique(labels)
        matches, no_matches = dict(), dict()
        for x in ulabels:
            matches[x] = np.where(labels == x)[0]
            no_matches[x] = np.where(labels != x)[0]

        candidates = np.random.randint(0, len(labels), size=num_triplets)
        candidates = labels[candidates]
        for x in candidates:
            while len(matches[x]) < 2:
                x = candidates[np.random.randint(0, len(labels))]
            idx_a, idx_p = np.random.choice(matches[x], 2, replace=False)
            idx_n = np.random.choice(no_matches[x], 1)[0]
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)
    
    def next_train(self):
        while True:
            if self.cur_train_index >= len(self.triplets):
                self.cur_train_index = 0
                self.X_train, y_train = self.train_generator.next()
                self.triplets = self.generate_triplets(y_train, self.num_train_triplets)
            triplet_indices = self.triplets[self.cur_train_index : self.cur_train_index + self.batch_sz]
            self.cur_train_index += self.batch_sz
            yield ([ self.X_train[triplet_indices[:, 0]], self.X_train[triplet_indices[:, 1]], 
                  self.X_train[triplet_indices[:, 2]] ], np.ones(len(triplet_indices)))

    def next_val(self):
        while True:
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index= 0
            triplet_indices = self.val_triplets[self.cur_val_index : self.cur_val_index + self.batch_sz]
            self.cur_val_index += self.batch_sz
            yield ([ self.X_val[triplet_indices[:, 0]], self.X_val[triplet_indices[:, 1]], 
                  self.X_val[triplet_indices[:, 2]] ], np.ones(len(triplet_indices)))