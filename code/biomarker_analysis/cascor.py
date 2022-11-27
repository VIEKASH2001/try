#Code adopted from: https://github.com/ephe-meral/cascor 
# https://towardsdatascience.com/cascade-correlation-a-forgotten-learning-architecture-a2354a0bec92


import numpy as np
from numpy.random import Generator, PCG64

import torch

import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('qtagg')


rnd = Generator(PCG64(12345))
# To be used like: numpy.random, e.g. rnd.random(2, 3) produces a random (2,3) array
# We'll use this to generate initial weights for our neurons



# Param shapes: x_: (n,i), y_: (n,o), weights: (i,o)
#   Where n is the size of the whole sample set, i is the input count, o is the output count
#   We expect x_ to already include the bias
# Returns: trained weights, last prediction, last iteration, last loss
# NB: Differentiation is done via torch
def quickprop(x_, y_, weights,
              activation=torch.nn.Sigmoid(),
              loss=torch.nn.MSELoss(),
              learning_rate=1e-4,
              tolerance=1e-6,
              patience=20000,
              debug=False):
    # Box params as torch datatypes
    x = torch.Tensor(x_)
    y = torch.Tensor(y_)
    w = torch.Tensor(weights)

    # Keep track of mean residual error values (used to test for convergence)
    L_mean = 1
    L_mean_prev = 1
    L_mean_diff = 1
    
    # Keep track of loss and weight gradients
    dL = torch.zeros(w.shape)
    dL_prev = torch.ones(w.shape)
    dw_prev = torch.ones(w.shape)

    i = 0
    predicted = []

    # This algorithm expects the mean losses to converge or the patience to run out...
    while L_mean_diff > tolerance and i < patience:
        # Prep iteration
        i += 1
        dL_prev = dL.clone()
        # NB: The following can probably done better with torch.no_grad(), but I couldn't make it work
        w_var = torch.autograd.Variable(torch.Tensor(w), requires_grad=True)
        
        # Calc forward and loss
        predicted = activation(torch.mm(x, w_var))
        L = loss(predicted, y)
        
        # Keep track of losses and use as convergence criterion if mean doesn't change much     
        L_mean = L_mean + (1/(i+1))*(L.detach().numpy() - L_mean)
        L_mean_diff = np.abs(L_mean_prev - L_mean)
        L_mean_prev = L_mean
        
        # Calc differential and do the weight update
        L.backward()
        
        dL = w_var.grad.detach() # =: partial(L) / partial(W)
        dw = dw_prev * dL / (dL_prev - dL + 1e-10) # Prevent div/0
        
        dw_prev = dw.clone()
        
        w += learning_rate * dw
        
        if debug and i % 100 == 99:
            print("Residual           ", L.detach().numpy())
            print("Residual mean      ", L_mean)
            print("Residual mean diff ", L_mean_diff)
        
    return w.detach().numpy(), predicted.detach().numpy(), i, L.detach().numpy()

# The vector x is the values from the earlier hidden/input layers per each sample
# Parameter shapes: x - (n,i), y - (n,o)
#   Where n is the size of the whole sample set, i is the input count, o is the output count
#   We expect x_ to already include the bias
def train_outputs(x, y):
    # Next we need to create a weight vector with the right shape
    n, i = x.shape
    n, o = y.shape
    
    weights = rnd.uniform(-0.01, 0.01, size=(i, o))
    
    # And run through the training
    weights, predicted, i, loss = quickprop(x, y, weights)
    
    return weights, predicted


def train_hidden(x, y, predicted, debug=False):
    # Figure out how many weights we need
    n, i = x.shape
    
    # And initialize a weights matrix
    weights = torch.Tensor(rnd.uniform(-0.01, 0.01, size=(i, 1)))
    
    # Calculate the residuals for correlation
    err = torch.Tensor(y - predicted)
    err_mean = torch.mean(err, axis=0)
    err_corr = (err - err_mean)
    
    if debug:
        plt.imshow(err_corr.reshape(target1.shape), cmap='hot', interpolation='nearest')
        plt.show()
    
    # Create a custom loss function (S)
    def covariance(pred, target):
        pred_mean = torch.mean(pred, axis=0)
        # We want to try to maximize the absolute covariance, but quickprop will minimize its loss function
        # Therefore, we need to multiply by (-1) to guide the optimizer correctly
        loss = -torch.sum(torch.abs(torch.sum((pred - pred_mean)*(target), axis=0)), axis=0)
        return loss
        
    # Use quickprop to generate the weights based on the special loss function
    # We also need to pass in the residual errors as a target
    weights, predicted, i, loss = quickprop(x, err_corr, weights, loss=covariance)
    
    return weights, predicted


def create_trainset(target):
    idxs = np.asarray(list(np.ndindex(target.shape)))
    # Normalize inputs
    idxs = idxs / np.linalg.norm(idxs, axis=0)
    # Add bias vector:
    x = np.ones((idxs.shape[0], idxs.shape[1]+1))
    x[:,:-1] = idxs

    y = target.reshape((-1, 1))
    
    return x, y


def cascor_training(x, y):

    #Train initial output layer
    w, pred = train_outputs(x, y)

    #Train hidden layer
    neuron_w, neuron_value = train_hidden(x, y, pred, debug=False)

    #Combining Hidden & Output Neurons
    x2 = np.concatenate((x, neuron_value), axis=1)

    w2, pred2 = train_outputs(x2, y)



if __name__ == "__main__":

    target1 = np.concatenate((np.zeros((20,40)), np.ones((20,40))), axis=0)
                            
    plt.imshow(target1, cmap='hot', interpolation='nearest')
    plt.show()

    target2 = np.concatenate(
                    (np.concatenate((np.zeros((20,20)), np.ones((20,20))), axis=1),
                    np.ones((20,40))), 
                axis=0)
                            
    plt.imshow(target2, cmap='hot', interpolation='nearest')
    plt.show()

    x, y = create_trainset(target1)
    w, pred = train_outputs(x, y)

    plt.imshow(pred.reshape(target1.shape), cmap='hot', interpolation='nearest')
    plt.show()


    x, y = create_trainset(target2)
    w, pred = train_outputs(x, y)

    plt.imshow(pred.reshape(target1.shape), cmap='hot', interpolation='nearest')
    plt.show()

    neuron_w, neuron_value = train_hidden(x, y, pred, debug=True)

    plt.imshow(neuron_value.reshape(target1.shape), cmap='hot', interpolation='nearest')
    plt.show()


    x2 = np.concatenate((x, neuron_value), axis=1)

    w2, pred2 = train_outputs(x2, y)

    plt.imshow(pred2.reshape(target1.shape), cmap='hot', interpolation='nearest')
    plt.show()

