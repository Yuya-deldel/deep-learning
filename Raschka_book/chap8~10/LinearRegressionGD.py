import numpy as np

class LinearRegression:       # full batch 
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                      # learning rate 
        self.n_iter = n_iter                # maximum iteration number of times
        self.random_state = random_state    # random seed for initialization of weight parameter 

    def fit(self, X, y):        # X: feature data, y: target data
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])     # initialize by small random number
        self.b_ = np.array([0.])                                        # initialize by 0.0
        self.losses_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output 
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self 
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return self.net_input(X)