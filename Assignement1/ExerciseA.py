import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


######################### Reading the file m_cerditcard.csv ###########################

creditcardTransactions = pd.read_csv('m_creditcard.csv')
features = creditcardTransactions[['V2', 'V11']]
isFraud = creditcardTransactions['Class']

######################### preceptron implementation ###########################

def preceptron(X, Y, learningRate, iterations):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)

    inputMatrix = np.c_[np.ones(inputMatrix.shape[0]), inputMatrix]

    weightVector = np.random.rand(inputMatrix.shape[1])
    weightHistory = weightVector
    errorHistory = []
    numberOfIterations = 0
    output  = []

    for iter in np.arange(0,iterations):
        error = 0
        for i in np.arange(0,inputMatrix.shape[0]):
            instance = inputMatrix[i]
            h = instance.dot(weightVector)
            output = 0 if(h < 0.5) else 1
            error += 1 if(output != targetOuput[i]) else 0

            weightVector -= learningRate*(output-targetOuput[i])*instance

            weightHistory = np.c_[weightHistory, weightVector]
            errorHistory = np.r_[errorHistory, error]

    return weightVector,inputMatrix,weightHistory,errorHistory


######################### Training ###########################
weightVector, X,weightHistory,errorHistory = preceptron(features,isFraud,0.8,50)

######################### Calculating the output ###########################
output = X.dot(weightVector)

######################### Plotting ###########################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(features[['V11']][isFraud==0],features[['V2']][isFraud==0], isFraud[isFraud==0], color='b')
ax.scatter(features[['V11']][isFraud==1], features[['V2']][isFraud==1], isFraud[isFraud==1], color='r')
ax.plot(features['V11'],features['V2'],output)

ax.set_xlabel('V11')
ax.set_ylabel('V2')
ax.set_zlabel('Fraud')

fig1, ax1 = plt.subplots()


ax1.scatter(np.arange(0,weightHistory.shape[1]),weightHistory[1,0:], color='b')
ax1.scatter(np.arange(0,weightHistory.shape[1]),weightHistory[2,0:], color='r')

plt.show()