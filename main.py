import numpy as np
from functions import relu


def forward_pass(x:float, W:np.array, b:np.array):

    layer_number = np.size(W, axis=0)
    h = np.zeros_like(b)

    # Hidden layer(s)
    for i in range(layer_number-1):                              # so iterates over 1st to (n-1)th layer - i.e. hidden layers
        h[i] = relu(np.dot(W[i],x) + b[i])

    # Output layer
    b_final = 0.0
    y = np.dot(W[layer_number-1], h[layer_number-2]) + b_final   # n-th layer (output layer)

    print(b.shape)
    print(h.shape)
    print(W.shape)
    return y


def loss(x:float, y_est:float):
    y = x**2                 # true 'y' I being compared against
    return (y - y_est)**2



x_values = [-2,-1,0,1,2]
W = np.array([[1.0,-1.0],
             [1.0,1.0]])
b = np.array([[0.0,0.0]]) # setting this as [[]] so that elements are arrays rather than scalars

for x in x_values:
    print(forward_pass(x,W,b))


