import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


######################### Reading the file m_cerditcard.csv ###########################

creditcardTransactions = pd.read_csv('m_creditcard.csv')
features = creditcardTransactions[['V2', 'V11']]
isFraud = creditcardTransactions['Class']

x = np.array([1,2,3,4])
y = np.array([[5,6,7,8],[1,1,1,1]])
# print(features)



######################### preceptron implementation ###########################

def preceptron(X, Y, learningRate, iterations):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)

    inputMatrix = np.c_[np.ones(inputMatrix.shape[0]), inputMatrix]

    weightVector = np.random.rand(inputMatrix.shape[1])
    weightHistory = weightVector
    errorHistory = []
    numberOfIterations = 0
    while True:
        h = inputMatrix.dot(weightVector)
        h[np.where(h >= 0.5)] = 1
        h[np.where(h < 0.5)] = 0
        activationOutput = h
        error = inputMatrix.T*((activationOutput-targetOuput)).T
        weightDelta = (1.0/inputMatrix.shape[0])*error.sum(axis=1)
        weightVector = weightVector - learningRate*weightDelta
        weightHistory = np.c_[weightHistory, weightVector]
        numberOfIterations +=1
        if sum(abs(np.subtract(weightHistory[0:,weightHistory.shape[1]-2], weightVector))) < 0.0001:
            break

    # for iter in np.arange(0,iterations):
    #     error = 0
    #     for i in np.arange(0,inputMatrix.shape[0]):
    #         instance = inputMatrix[i]
    #         h = instance.dot(weightVector)
    #         output = 0 if(h < 0.5) else 1
    #         error += 1 if(output != targetOuput[i]) else 0

    #         weightVector -= learningRate*(output-targetOuput[i])*instance

    #         weightHistory = np.c_[weightHistory, weightVector]
    #         errorHistory = np.r_[errorHistory, error]

    # return weightVector,inputMatrix,weightHistory,errorHistory
    return weightVector,inputMatrix,weightHistory,numberOfIterations


######################### Training ###########################
weightVector, X,weightHistory,numberOfIterations = preceptron(features,isFraud,0.2,50)
# print(preceptron(features,isFraud,0.8,200))


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

print(weightHistory.shape[1],numberOfIterations)
ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[1,0:], color='b')
ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[2,0:], color='r')

plt.show()