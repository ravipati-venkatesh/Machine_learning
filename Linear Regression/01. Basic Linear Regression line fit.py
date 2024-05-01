#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import time


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(0)
X =  np.random.rand(100, 1)
y = 9 + 7 * X +  np.random.randn(100, 1)
weights = np.random.randn(2,1)
X_ = np.c_[np.ones((100, 1)), X]


# In[32]:


plt.scatter(X, y)


# In[33]:


X_.shape


# In[38]:


def gradient_descent(X_, y, weights, learning_rate=0.001):
    gradients = 2*X_.T.dot(X_.dot(weights)-y)
    weights = weights - learning_rate * gradients
    return weights


# In[ ]:


iterations = 2000
for i in range(0, iterations):
    weights = gradient_descent(X_, y, weights)
    if i%100==0:
        print(i, weights)
    plt.scatter(X, y)
    y_predict = X_.dot(weights)
    plt.plot(X, y_predict)
    plt.title('Convergence of Regression Line in Each Epoch using Gradient Descent')
    plt.show()
    time.sleep(0.001)


# In[ ]:




