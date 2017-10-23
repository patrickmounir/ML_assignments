from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('timesereis_8_2.csv')
features = data[['0', '1','2', '3','4', '5','6', '7']]
target=data[['8','9']]

#####################MLP######################################
def MLP(X, Y, learningRate,layers):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)
    reg =MLPRegressor(solver='sgd',hidden_layer_sizes=layers,learning_rate='constant',learning_rate_init=learningRate)
    print(reg.predict(np.array([2.53,2.3,1.91,1.82,1.86,1.96,1.912,2.98]).reshape(1, -1)))
    return reg.fit(inputMatrix,targetOuput)

############################Main#############################
reg = MLP(features,target,0.001,(10,8))