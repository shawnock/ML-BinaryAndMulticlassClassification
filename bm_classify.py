import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    
    Note: 
    - find the gradients of the two loss functions by yourself and 
    apply average gradient descent to update  𝑤,𝑏  in each iteration
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0


    b = 0
    if b0 is not None:
        b = b0

    # since y is {0, 1}, need to change it to {-1, 1}
    y[y == 0] = -1

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        
        # Z = y_n(w^T * x_n+b)
        
        # check if Z <= 0
        # if so, gradient of w = y_n * x_n, else gradient of w = 0
        # if so, gradient of b = y_n, else gradient of b = 0

        # add b inside of w
        new_w = np.zeros(D + 1)

        # add a column in X to make balance
        X = np.insert(X, 0, 1, 1)

        for i in range(max_iterations):
            indicator_check = np.where(y * (np.dot(X, new_w)) <= 0, 1, 0)

            new_w = new_w + (np.dot(indicator_check * y, X)) / N * step_size

        w = new_w[1:]
        b = new_w[0]


        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #

        # add b inside of w
        new_w = np.zeros(D + 1)

        # add a column in X to make balance
        X = np.insert(X, 0, 1, 1)

        for i in range(max_iterations):
            new_w += np.dot(sigmoid(-(y * np.dot(new_w, X.transpose()))) * y, X) / N * step_size

        w = new_w[1:]
        b = new_w[0]
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1/(1+ np.exp(-z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #


        temp = np.dot(w, X.transpose()) + b

        preds = np.where(temp < 0, 0, 1)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        temp = sigmoid(np.dot(w, X.transpose()) + b)

        preds = np.where(temp > 0.5 , 1, 0)
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        for i in range(max_iterations):
            # pick a random
            x_n = np.random.choice(range(N))
            p = softmax(x_n, w, b)

            if i == 0:
                print("test below:")
                print(p)

            #update w
            w_gradient = np.dot(p.transpose(), x_n)
            w = w - step_size * w_gradient


            # update b
            b_gradient = np.sum(p,axis=0)
            b = b - step_size * b_gradient


        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        for i in range(max_iterations):
            p = softmax(X, w, b)
            p[np.arange(N), y] -= 1

            if i == 0:
                print("p after:")
                print(p)

            # update w
            w_gradient = np.dot(p.transpose(), X)
            w = w - step_size/N * w_gradient

            # update b
            b_gradient = np.sum(p,axis=0)
            b = b - step_size/N * b_gradient

        ############################################


    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    temp = np.dot(X,w.T) + b.reshape(1,-1)

    preds = np.argmax(temp, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds

def softmax(X, w, b):

    numerator = np.dot(X, w.transpose()) + b.reshape(1, -1) - np.max(np.dot(X, w.transpose()) + b.reshape(1, -1), axis=1).reshape(-1, 1)
    numerator = np.exp(numerator)

    denominator = np.sum(numerator, axis = 1)

    p = numerator / denominator.reshape(-1,1)

    return p




        