'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-06-04
 *  Modified On: 2020-06-04
 '''
import pandas as pd
import numpy as np
import random

# Randoms
def random_float(low,high):
    return low + (high - low) * random.random()

# Uniform Distribution
def uniform(a,b):
    return random_float(min(a,b),max(a,b))

# Normal Distribution
def normal(mean = 0.0, sigma = 1.0):
    u = uniform(0.0,1.0)
    v = uniform(0.0,1.0)
    x = np.math.sqrt(-2*np.math.log(u))*np.math.cos(2*np.math.pi*v)
    #y= np.math.sqrt(-2*np.math.log(u))*np.math.sin(2*np.math.pi*v)
    return mean + sigma * x

# Truncated Normal Distribution
def truncated_normal(mean, sigma):
    while True:
        x = normal(mean,sigma)
        if x >= -2*sigma and x <= 2*sigma:
            return x

# Log-normal Distribution
def lognormal(mean = 0.0, sigma = 1.0):
    return np.math.exp(normal(mean,sigma))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


class Algo:
    #----------------------------------------------------------
    # Constructor
    #----------------------------------------------------------
    def __init__(self,activation,n_inputs,initializer="normal"):
        self.f = activation
        # Weights Initialization
        if initializer == "zero":               # Zero Initialization
            self.w = np.array([0.0 for i in range(n_inputs)])
            self.b = 0.0
        if initializer == "uniform":            # Uniform Initialization
            self.w = np.array([uniform(0,1) for i in range(n_inputs)])
            self.b = uniform(0,1)
        if initializer == "normal":             # Normal (Gaussian) Initialization
            self.w = np.array([normal(0,1) for i in range(n_inputs)])
            self.b = normal(0,1)
        if initializer == "truncated_normal":   # Truncated Normal Initialization
            self.w = np.array([truncated_normal(0,1) for i in range(n_inputs)])
            self.b = normal(0,1)
        if initializer == "lognormal":          # Log Normal Initialization
            self.w = np.array([lognormal(0,1) for i in range(n_inputs)])
            self.b = normal(0,1)
        if initializer == "glorot_uniform":     # Glorot (Xavier) Uniform Initialization
            self.w = np.array([uniform(0,1) for i in range(n_inputs)]) / (n_inputs + 1)
            self.b = uniform(0,1)
        if initializer == "glorot_normal":      # Glorot (Xavier) Normal Initialization
            self.w = np.array([normal(0,1) for i in range(n_inputs)]) / (n_inputs + 1)
            self.b = normal(0,1)
