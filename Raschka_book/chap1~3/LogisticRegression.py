from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression:       # full batch 
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                      # learning rate 
        self.n_iter = n_iter                # maximum iteration number of times
        self.random_state = random_state    # random seed for initialization of weight parameter 

    def fit(self, X, y):        # X: feature data, y: target data
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])     # initialize by small random number
        self.b_ = np.float_(0.)                                         # initialize by 0.0
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output 
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - (1.0 - y).dot(np.log(1.0 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self 
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)