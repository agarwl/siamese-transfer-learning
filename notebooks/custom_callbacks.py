import numpy as np
from keras.callbacks import Callback

class LossHistory(Callback):
    
    def __init__(self, X_val, y_val, alphabets={}):
        self.X_val = X_val
        self.y_val = y_val
        self.alphabets = alphabets
        self.matches = {}
        for x in np.unique(y_val):
            self.matches[x] = np.where(y_val == x)[0]
        self.true_indices = np.arange(20)
        self.one_shot_indices = list(self.one_shot_task(alph) for alph in self.alphabets 
                   for _ in range(2))
        self.siamese_net = None

    def on_epoch_end(self, epoch, logs={}):
        self.siamese_net = self.model.get_layer('siamese_net')
        one_shot_acc = self.one_shot_acc()
        logs['oneshot_acc'] = one_shot_acc
        print(" oneshot_acc - {}".format(one_shot_acc))
        
    def one_shot_acc(self):
        acc = [self.compute_acc(index[0], index[1]) for index in self.one_shot_indices]
        return np.mean(acc)
    
    def compute_acc(self, support, test):
        X_support, X_test = self.X_val[support], self.X_val[test]
        class_indices = self.compute_pred_class(X_test, X_support)
        return (np.sum(class_indices == self.true_indices))/20.0

    def one_shot_task(self, alph):
        class_arr = self.alphabets[alph]
        sample_classes = np.random.choice(class_arr, 20, replace = False)
        train_arr, test_arr = [], []
        drawers = np.random.choice(20, 2, replace = False)
        support_arr = [self.matches[x][drawers[0]] for x in sample_classes] 
        test_arr = [self.matches[x][drawers[1]] for x in sample_classes]
        return (support_arr, test_arr)
        
    def kernel(self, x, y):
        return self.siamese_net.predict([x, y]).ravel()

    def compute_pred_class(self, X, Y):
        n = Y.shape[0]
        columns = (np.array([x] * n) for x in X)    
        pred_classes = np.array([np.argmax(self.kernel(col, Y)) for col in columns])
        return pred_classes