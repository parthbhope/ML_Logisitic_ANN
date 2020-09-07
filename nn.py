import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time

class NerualNetwork(object):
    def __init__(self,layer_dims,mode='gaussian'):
        np.random.seed(1)
        self.parameters = {}
        L = len(layer_dims) 
        if mode == 'gaussian':
            for l in range(1, L):
                self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
                self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        else:
            for l in range(1, L):
                self.parameters['W' + str(l)] = np.random.uniform(low=-0.6,high=0.6,size=(layer_dims[l], layer_dims[l-1]))
                self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(self.parameters['b' + str(l)].shape == (layer_dims[l], 1))
            
    def sigmoid(self,Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self,Z):
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache


    def relu_backward(self,dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ

    def sigmoid_backward(self,dA, cache): 
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    
    def linear_forward(self,A, W, b):
        Z = W.dot(A) + b
    
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
    
        return Z, cache    
    

    def linear_activation_forward(self,A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def linear_backward(self,dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1./m * np.dot(dZ,A_prev.T)
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self,dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db
    
    def L_model_forward(self,X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)

        assert(AL.shape == (1,X.shape[1]))

        return AL, caches
    
    def compute_cost(self,AL, Y):
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())

        return cost
    
    def L_model_backward(self,AL, Y, caches):
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation = "sigmoid")

        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    
    def update_parameters(self,parameters, grads, learning_rate):
        L = len(parameters) // 2 # number of layers in the neural network
        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters
    
    def predict(self,X, y,ret_out=False):
        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))

        # Forward propagation
        probas, caches = self.L_model_forward(X, self.parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
#         print("Accuracy: "  + str(np.sum((p == y)/m))) 
        if ret_out:
            return p,np.sum((p == y)/m)
        else:
            return np.sum((p == y)/m)
            
    def train(self,X,Y,learning_rate = 0.07, num_iterations = 2000, print_cost=False):
        np.random.seed(1)
        costs = []                         # keep track of cost
        accuracy = []
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X, self.parameters)

            # Compute cost.
            cost = self.compute_cost(AL, Y)

            # Backward propagation.
            grads = self.L_model_backward(AL, Y, caches)

            # Update parameters.
            parameters = self.update_parameters(self.parameters, grads, learning_rate)
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
                accuracy.append(self.predict(X,Y))

        # plot the cost
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(1,3,1)
        ax1.plot(np.squeeze(accuracy))
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('iterations (per hundreds)')
        ax1.title.set_text("Learning rate =" + str(learning_rate))
        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(np.squeeze(costs))
        ax2.set_ylabel('cost')
        ax2.set_xlabel('iterations (per hundreds)')
        ax2.title.set_text("Learning rate =" + str(learning_rate))
        plt.show()
                            