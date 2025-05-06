from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt
import numpy as np

class ADALINE_full_batch:       # ADAptive LInear NEuron 
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
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self 
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        return X        # Id 
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
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

    def plot_losses(self, logplot=0):
        if logplot == 0:
            plt.plot(range(1, len(self.losses_) + 1), self.losses_, marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Mean squared error')
            plt.title('Adaline - Learning rate 0.0001')
        elif logplot == 1:
            plt.plot(range(1, len(self.losses_) + 1), np.log10(self.losses_), marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('log(Mean squared error)')
            plt.title('Adaline - Learning rate 0.1')
        
        plt.show()

################################################################################################
class ADALINE_sgd:      # ADAptive LInear NEuron - Stochastic Gradient Descent
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta                      # learning rate 
        self.n_iter = n_iter                # maximum iteration number of times
        self.shuffle = shuffle
        self.random_state = random_state    # random seed for initialization of weight parameter 

    def fit(self, X, y):        # X: feature data, y: target data
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            
            self.losses_.append(np.mean(losses))
        return self 
    
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self 
    
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output 
        self.w_ += self.eta * 2.0 * xi * error 
        self.b_ += self.eta * 2.0 * error 
        loss = error**2
        return loss

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        return X        # Id 
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
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


    def plot_losses(self, logplot=False):
        if not logplot:
            plt.plot(range(1, len(self.losses_) + 1), self.losses_, marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Mean squared error')
            plt.title('Adaline - Learning rate 0.0001')
        else:
            plt.plot(range(1, len(self.losses_) + 1), np.log10(self.losses_), marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('log(Mean squared error)')
            plt.title('Adaline - Learning rate 0.1')
        
        plt.show()