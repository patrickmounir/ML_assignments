import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################### Reading the file m_cerditcard.csv ###########################

flights = pd.read_csv('HA.csv')
flights = flights[['DISTANCE', 'ELAPSED_TIME']].dropna()
Distance = flights['DISTANCE']
ElapsedTime = flights['ELAPSED_TIME']

print(Distance)
######################### Divide Training and Testing Implementation ###########################

def trainingTest(X, Y, trainingPercentage):
    X = np.array(X)
    Y = np.array(Y)
    trainingLength = int(X.shape[0]*trainingPercentage)
    xTrain = np.zeros(trainingLength)
    yTrain = np.zeros(trainingLength)
    xTest = np.zeros(X.shape[0]-trainingLength)
    yTest = np.zeros(X.shape[0]-trainingLength)
    for i in np.arange(0,trainingLength):
        xTrain[i] = X[i]
        yTrain[i] = Y[i]
    for i in np.arange(trainingLength, X.shape[0]):
        xTest[i-trainingLength] = X[i]
        yTest[i-trainingLength] = Y[i]
        

    return xTrain,xTest,yTrain,yTest
######################### Mean Square Error Implementation ###########################

######################### Preceptron Implementation ###########################

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
            output = h
            error += 1 if(output != targetOuput[i]) else 0

            weightVector -= learningRate*(output-targetOuput[i])*instance

            weightHistory = np.c_[weightHistory, weightVector]
            errorHistory = np.r_[errorHistory, error]

    return weightVector,inputMatrix,weightHistory,errorHistory


######################### Training ###########################

xTrain,xTest,yTrain,yTest = trainingTest(Distance, ElapsedTime,0.8)
print(yTrain)
# weightVector, X, weightHistory, errorHistory = preceptron(xTrain,ytrain,0.8,50)
######################### Testing ###########################


######################### Plotting ###########################

fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()

ax.scatter(Distance, ElapsedTime, marker='x')


# ax1.scatter(np.arange(0,weightHistory.shape[0]),weightHistory[0:,1], color='b')
# ax1.scatter(np.arange(0,weightHistory.shape[0]),weightHistory[0:,2], color='r')

plt.show()