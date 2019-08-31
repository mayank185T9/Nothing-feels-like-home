import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Line(test_x, m, c):
    y_value = m*test_x + c
    return y_value

def MSECost(y,x,m,c):
    cost = 0
    for i in range(len(y)):
        cost = cost + (y[i]- Line(x[i],m,c))**2
        cost = cost/len(y)
    print(cost)
    return cost
    
#print(MSECost(Y,X,0.255826,0.003871))

def Grad(y,x,m,c,lr):
    dm = 0
    dc = 0
    for i in range(len(y)):
        dm = dm + x[i]*(y[i]- Line(x[i],m,c))
        dc = dc + (y[i]- Line(x[i],m,c))
    mean_dm = -2*dm/len(x)
    mean_dc = -2*dc/len(x)
    m = m - lr*mean_dm
    c = c - lr*mean_dc
    print(m,c)
        
    return mean_dm, mean_dc
    
def OptimizeSampleFullDataset(epochs, m, c,thld, x, y, lr):
    for i in range(epochs):
        cost=MSECost(y,x,m,c) 
        if (((cost) < thld)):
            break
        else:
            #a.append(cost)
            dm,dc = Grad(y,x,m,c,lr)
            m -=dm*lr
            c -=dc*lr
            a.append(MSECost(y,x,m,c))
    return m,c,a

lr = 0.0000001
epochs = 6000
M = 0
C = 0
finalM, finalC, plotCosts = OptimizeSampleFullDataset(epochs, M, C,0.1, X, Y, lr)
finalM,finalC

plt.plot(X, [xi*finalM + finalC for xi in X])
plt.scatter(X,Y)

plt.plot(X, [x*2 for x in range(len(X))])
plt.show()
