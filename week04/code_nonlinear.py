import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import urllib


def load_data():
    """ Simple helper function for downloading and loading data """
    print('loading data for nonlinear experiment')
    filename = 'nonlinear_data.npz'
    if not os.path.exists(filename):
        filepath = 'https://github.com/gronlund/Au_ml19/raw/master/week4/nonlinear_data.npz'
        print('file not exists - downloading:', filepath)
        with open(filename, 'wb') as fh:
            fh.write(urllib.request.urlopen(filepath).read())
    D =  np.load(filename, allow_pickle=True)
    return D


class PerceptronClassifier():

    def __init__(self):
        self.w = None
        
    def fit(self, X, y, maxiter=1 << 16, w=None):
        """
        Implement Pocket Perceptron learning algorithm - run for at most maxiter iterations and store best w found as well as the training history 
        
        Args:
        X: numpy array shape (n,d) - training data 
        y: numpy array shape (n,) - training labels
        maxiter: int, maximum number of iterations to run
        w: numpy array shape (d,) - inital w if desired
        Saves:
        w: numpy array shape (d,) normal vector of the best hyperplane found for separating X, y 
        may not seperate the data fully
        hist: list of (w, x, y) - algorithm history - store current w and misclassified point x and label y picked for update in each round of the algorithm. 
        Used only for animation - you can ignore if you do not need animation (remember to take a copies)
        """
        if w is None:
            w = np.zeros(X.shape[1])
        bestw = w
        bestscore = 0
        L = []
        # L.append((w.copy(), X.copy(), y.copy(), bestscore)) # to store current update (before w is updated)
        ### YOUR CODE

        # Find an xi for which sign(w^Txi) != yi
        # change w by w += yi*xi
        # if this improved the overall score of the perceptron, save w and append w, xi, yi, to the history
        import random

        self.w = w
        best_score = 0
        score = 0

        def get_misclassified():
            misses = np.where(self.predict(X) != y)[0]
            if len(misses) == 0:
                return None
            else:
                return random.choice(misses)

        for i in range(maxiter):
            index = get_misclassified()
            if index is None:
                score = 1.0
                break

            L.append((self.w.copy(), X[index].copy(), y[index].copy(), score))

            self.w += X[index, :] * y[index]
            score = self.score(X, y)
            if score > best_score:
                best_score = score
                bestw = self.w

        ### END CODE
        L.append((self.w.copy(), None, None, score))  # to store final w
        self.w = bestw
        self.history = L

    def predict(self, X):
        """ predict function for classifier
        Args:
          X (numpy array,  shape (n,d))
        Returns
          pred (numpy array,  shape(n,))
        """
        pred = None
        ### YOUR CODE HERE 1-2 lines
        """
               sign(w^T@X^T)
            sign of w^T@X_i forall i
                                  |x0,0, x0,1, ... x0,d|T
            [w0, w1, ..., wn]^T @ |x1,0, x1,1, ... x1,d|
                                  |...   ...   ... ... |
                                  |xn,0, xn,0, ... xn,d|
            -----------------------------------------------------------------------------
            [(x0,0*w0 + x0,1*w1 +...+ x0,d*wn), ... , (xn,0*w0 + xn,1*w1 +...+ xn,d*wn)]
            =============================================================================
        """

        #pred = np.sign(np.array(self.w).T @ np.array(X).T)
        pred = np.sign(X @ self.w)

        ### END CODE
        return pred

    def score(self, X, y):
        """ Return accuracy of model on data X with labels y
        
        Args:
          X (numpy array shape n, d)
        returns
          score (float) classifier accuracy on data X with labels y
        """
        score = 0
        ### YOUR CODE HERE 1-3 lines
        score = (self.predict(X) == y).mean()
        ### END CODE
        return score
    
        


def test_pla_train(n_samples=10):
    """ Test function for pla train """
    from sklearn.datasets import make_blobs
    print('Test perceptron classifier on random blobs - should be very linearly separable')
    centers = [(-50, -50), (50, 50)]
    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
    y = y * 2 - 1
    classifier = PerceptronClassifier()
    classifier.fit(X, y)
    assert np.all(classifier.predict(X) == y), 'all predictions should be correct, and they are not'
    print('Test completed succesfully')
    

def make_hyperplane(w, ax):
    """ 
    Make the hyperplane (line) w0 + w1*x1 + w2*x2 = 0 in the range R = [xmin,xmax] times [ymin,ymax] for a generic w = (w0, w1, w2) for plotting
    
    Copy from last week,
    Remember to handle possible special cases! 
    
    Notice how we pass along optional arguments to the plot function, which allows us to change color, etc. of the hyperplanes.

    Args:
    w: numpy array shape (d,)
    ax: matplotlib axes object to plot on
    
    """

    if w[1] == 0 and w[2] == 0:
        print('Invalid hyperplane')
        return None, None
        # Notice that w1 and w2 are not allowed to be 0 simultaneously, but it may be the case that one of them equals 0

    xmin, xmax, ymin, ymax = ax.axis()

    # Write the code here to create two NumPy arrays called x and y.
    # The arrays x and y will contain the x1's and x2's coordinates of the two endpoints of the line, respectively.

    x = np.array((0, 1))
    y = np.array((0, 1))

    ### YOUR CODE HERE 4-8 lines
    if not w[1]:
        x = np.array([-(w[0] / w[2]) - ((xmin * w[1]) / w[2]), -(w[0] / w[2]) - ((xmin * w[1]) / w[2])])
        y = np.array([xmin, xmax])
        # print("w1 = {0}, x = {1}, y = {2}".format(w[1],x,y))
    elif not w[2]:
        x = np.array([ymin, ymax])
        y = np.array([-(w[0] / w[1]) - ((ymin * w[2]) / w[1]), -(w[0] / w[1]) - ((ymin * w[2]) / w[1])])
        # print("w2 = {0}, x = {1}, y = {2}".format(w[2],x,y))
    else:
        x = np.array([xmin, -(w[0] / w[1]) - ((ymin * w[2]) / w[1])])
        y = np.array([-(w[0] / w[2]) - ((xmin * w[1]) / w[2]), ymin])
        # print("w0 = {0}, w1 = {1}, w2 = {2}, x = {3}, y = {4}".format(w[0],w[1],w[2],x,y))
    ### END CODE

    return x, y
    

def square_transform(X):
    """
    Implement the square transform including the bias variable 1 in the first column i.e 
    phi(x_1,x_2) = (1, x_1^2, x_2^2). 
    We will prepend a column of ones for the bias variable
    
    np.c_, np.transpose may be useful
    To raise a number to second power use operator ** i.e. 3**2 is 9
    
    As an example of what the function should do:
    >>> X = np.array([[1,2],[3,4]])
    >>> square_transform(X)
    array([[  1.,   1.,   4.],
       [  1.,   9.,  16.]])

    Args:
      X: np.array of shape (n, 2)

    Returns
      Xt: np.array of shape(n, 3) 
    """
    # Insert code here to transform the data - aim to make a vectorized solution!
    Xt = X

    ### YOUR CODE HERE 2-4 lines
    Xt = np.c_[np.ones(X.shape[0]), np.power(X, 2)]#(X @ np.transpose(X))]
    ### END CODE
    
    return Xt
    
def plot_contour(w, phi, ax):
    """
    Make a contour plot showing the decision boundary in the original input space R^2,
    which is a line in the feature space defined by phi

    Make a solution with for loops.        

    Args: 
     w: np.array shape (d,) the decision boundary vector
     phi: function, the transform phi to contour plot (function from (n, 2) array to (n, d) array
     ax: matplotlib Axes, to plot the contour on     
    """

    nsize = 100
    xs = ys = np.linspace(-1, 1, nsize)
    img = np.zeros((nsize, nsize)) # makes a 100 x 100 2d array
    ### YOUR CODE
    for i in range(len(xs)):
        for j in range(len(ys)):
            x = xs[i]
            y = ys[j]
            f = phi(np.array([[x, y]]))[0]
            img[i, j] = w @ f

    ### END CODE
    cont = ax.contour(xs, ys, img, [0], colors='r', linewidths=3)
    return cont

    
def poly_transform(X):
    """
    Compute the polynomial transform [x_1^i * x_i^j] for 0 <= i+j <=3
    List comprehensions may be very useful
    Also, np.c_ or np.transpose could be useful
    
    Args:
     X: numpy array shape (n, 2)

    Returns: 
      numpy arrays shape (n, d) 
    """
    Xt = []
    ### YOUR CODE HERE
    #size = 4
    #for x in X:
    #    t = []
    #    for i in range(size):       # [0,4)
    #        for j in range(size-i): # [0,4-i)
    #            t.append(np.multiply(np.power(x[0], i), np.power(x[1], j)))
    #    Xt.append(t)
    #Xt = np.array(Xt)
    for x in X:
        Xt.append(
            [
                1,
                x[0],
                x[1],
                np.power(x[0], 2),
                np.power(x[1], 2),
                np.power(x[0], 3),
                np.power(x[1], 3),
                x[0] * x[1],
                x[0] * np.power(x[1], 2),
                x[1] * np.power(x[0], 2)
            ]
        )
    # print(Xt)
    ### END CODE
    return np.array(Xt)



def plot_data():
    """
    Insert code to plot the data sets here
    
    plot_data using scatter plots in xrange and yrange set to [-1, 1] and set title to Data Set i for the i'th data set.
    We have created the 4 axes for you to plot on. ie axes[0] should contain data set 1 and so on.
    Plot the data on these 4.

     ax.scatter, ax.set_title may come in usefull
    """
    D = load_data()

    fig, axes = plt.subplots(1, 4, figsize=(12, 10))
    
    ### YOUR CODE 
    ### END CODE
    plt.show()

    
def plot_square_transform():
    """ Visualize the square transform """
    D = load_data()
    X = D['X1']
    y = D['y1']
    Xt = square_transform(X)
    cls = PerceptronClassifier()
    cls.fit(Xt, y)
    w_pla = cls.w 
    print('Hyperplane:', w_pla)
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
    axes[0].set_title('Data Set 1', fontsize=16)
    axes[1].scatter(Xt[:,1], Xt[:, 2], c=y, cmap=plt.cm.Paired, s=20)
    x, y = make_hyperplane(w_pla, axes[1])
    axes[1].plot(x, y, 'r--', linewidth=4)    
    axes[1].set_title('Data Set 1 - Transformed - Decision boundary', fontsize=16)
    plt.show()

    
def contour_test():
    """ Test contour algorithm """
    D = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    print('Contour for data set {0}'.format(1))
    X = D['X1']
    y = D['y1']
    Xt = square_transform(X)
    cls = PerceptronClassifier()
    cls.fit(Xt, y)
    w_pla = cls.w    
    ax = axes[0]        
    ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, s=20)
    plot_contour(w_pla, square_transform, ax)
    ax.set_xlim([-1, 1])
    ax.set_title('Data set {0}: Score: {1}'.format(0, cls.score(Xt, y)))

    fe = axes[1]
    fe.scatter(Xt[:, 1], Xt[:, 2], c=y, cmap=plt.cm.Paired, s=20)
    fe.set_xlim([0, 1])
    fe.set_title('Data set {0}: Feature Space')
    h1, h2 = make_hyperplane(w_pla, fe)
    fe.plot(h1, h2, 'r--', linewidth=4)
    plt.show()
    

def run():
    """
    For each data set
        # (1) apply the degree 3 polynomial transform.
        # (2) runs the perceptron learning algorithm.
        # (3) computes and prints the in sample error and plots the results including contour plots
    
    """
    D = load_data()
    fig, axes = plt.subplots(1, 4, figsize=(12, 10))
    for i in range(1, 5):
        X = D['X%d' % i]
        y = D['y%d' % i]
        Xt = poly_transform(X)
        cls = PerceptronClassifier()
        cls.fit(Xt, y)
        w_pla = cls.w
        ax = axes[i-1]
        ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, s=20)
        plot_contour(w_pla, poly_transform, ax)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_title("Nonlinear Perceptron - D{0}: Score: {1}".format(i, cls.score(Xt, y)))

        
    plt.show()

def run_linreg():
    """
    For each data set
        # (1) apply the degree 3 polynomial transform.
        # (2) runs the perceptron learning algorithm.
        # (3) computes and prints the in sample error and plots the results including contour plots
    
    """
    D = load_data()
    fig, axes = plt.subplots(1, 4, figsize=(12, 10))
    for i in range(1, 5):
        X = D['X%d' % i]
        y = D['y%d' % i]
        Xt = poly_transform(X)
        cls = LinRegClassifier()
        cls.fit(Xt, y)
        w_pla = cls.w
        ax = axes[i-1]
        ax.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, s=20)
        plot_contour(w_pla, poly_transform, ax)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_title("LinReg - D{0}: Score: {1}".format(i, cls.score(Xt, y)))
        
    plt.show()


class LinRegClassifier():

    def __init__(self):
        self.w = None
    
    def fit(self, X, y):
        """ 
        Linear Regression Learning Algorithm
        
        For this we compute the parameter vector         
        wlin = argmin ( sum_i (w^T x_i -y_i)^2 )    
        The pseudo-inverse operator pinv in numpy.linalg package may be useful, i.e. np.linalg.pinv

        Args:
        X: numpy array shape (n,d)
        y: numpy array shape (n,)
            
        Computes and stores w: numpy array shape (d,) the best weight vector w to linearly approximate the target from the features.

        """  
        w = np.zeros(X.shape[1])
        ### YOUR CODE HERE 1-3 lines
        ### END CODE
        self.w =  w

    def predict(self, X):
        """ predict function for classifier
        Args:
          X (numpy array,  shape (n,d))
        Returns
          pred (numpy array,  shape(n,))
        """
        pred = None
        ### YOUR CODE HERE 1-2 lines
        ### END CODE
        return pred

    def score(self, X, y):
        """ Return accuracy of model on data X with labels y
        
        Args:
          X (numpy array shape n, d)
        returns
          score (float) classifier accuracy on data X with labels y
        """
        score = 0 
        ### YOUR CODE HERE 1-3 lines
        ### END CODE
        return score

        
        

        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-platest', action='store_true', default=False)
    parser.add_argument('-plot', action='store_true', default=False)
    parser.add_argument('-square', action='store_true', default=False)
    parser.add_argument('-contour', action='store_true', default=False)
    parser.add_argument('-run', action='store_true', default=False)
    parser.add_argument('-linreg', action='store_true', default=False)
    
    
    args = parser.parse_args()
    if args.platest:
        test_pla_train()
    if args.plot:
        plot_data()
    if args.square:
        plot_square_transform()
    if args.contour:
        contour_test()
    if args.run:
        run()
    if args.linreg:
        run_linreg()
