from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta                      # learning rate 
        self.n_iter = n_iter                # maximum iteration number of times
        self.random_state = random_state    # random seed for initialization of weight parameter 

    # feature data: X, target data: y
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])     # initialize by small random number
        self.b_ = np.float_(0.)                                         # initialize by 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):        # update weight: w and bias: b
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi 
                self.b_ += update
                errors += int(update != 0.0)    # if update != 0.0, error += 1
                
            self.errors_.append(errors)     # number of error in each iteration

        return self 
    
    def net_input(self, X):     # z = w_vec * x_vec + b
        return np.dot(X, self.w_) + self.b_ 
    
    def predict(self, X):       # ^y = Heaviside(z)
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    ################################################################################     
    def plot_decision_regions(self, X, y, resolution=0.02):
        markers = ('o', 's', '^', 'v', '<')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        lab = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T) 
        lab = lab.reshape(xx1.shape)
        plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolors='black')
        plt.show()

    def plot_errors(self):
        plt.plot(range(1, len(self.errors_) + 1), self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.show()