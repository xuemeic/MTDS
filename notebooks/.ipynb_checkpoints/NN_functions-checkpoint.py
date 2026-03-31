import numpy as np

def relu(x):
    # relu function
    return np.maximum(x, 0)

def J_MSE(u, y):
    # MSE cost function: 1 / len(u) * ||u-y||^2
    N = len(u)
    return (np.sum((u - y)**2))/N

def simple_NN_pred(X_new, W1_trained, W2_trained):
    u1 = X_new@W1_trained
    h1 = relu(u1)
    u2 = h1@W2_trained
    return u2

def forward_prop(X, W1, W2):
    u1 = X@W1
    h = relu(u1)
    u2 = h@W2
    return u1, h, u2

def backward_prop(X, y, u1, h, u2, W1, W2):
    du2 = u2 - y
    dW2 = (h.T)@du2
    dh = du2@W2.T
    # du1 = dh*r'(u1): 
    # if u1(i,j) > 0, then r'(u1)(i,j)=1; if u1(i,j) <= 0, then r'(u1)(i,j)=0
    du1 = dh*relu(np.sign(u1)) # point-wise multiplication
    dW1 = X.T@du1
    return dW1, dW2
    
def simple_NN(X, y, seed = 2, lr = 1e-1, hidden_size = 5, max_iter = 30, **kwargs):
    '''
    X, y should be numpy arrays
    W1_init: optional argument for initialized weight
    W2_init: optional argument for initialized weight
    '''
    n, p = X.shape
    # reshape y to a column
    y = y.reshape([-1, 1])

    # size (number of nodes) of hidden layer
    p1 = hidden_size
    
    # random initialization if initial weights not provided
    np.random.seed(seed)
    w_scale = np.sqrt(6/(p+1))
    W1_rand = np.random.rand(p, p1) * w_scale 
    W2_rand = np.random.rand(p1, 1) * w_scale

    W1 = kwargs.get('W1_init', W1_rand)
    W2 = kwargs.get('W2_init', W2_rand)
    
    for i in range(max_iter):
        # forward propargation (usually should be a for loop, iterate over how many hidden layers there are)
        u1, h, u2 = forward_prop(X, W1, W2) # u2 is supposed to be approx y
    
        # compute cost
        J = J_MSE(u2, y)
    
        # backward propagation for computing gradient (usually a for loop iterate over # of hidden layers)
        dW1, dW2 = backward_prop(X, y, u1, h, u2, W1, W2)
        
        # gradient descent
        W1 = W1 - lr * dW1
        W2 = W2 - lr * dW2
    return W1, W2, J, u2