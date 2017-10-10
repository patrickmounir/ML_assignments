import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################### Reading the file m_cerditcard.csv ###########################

flights = pd.read_csv('HA.csv')
flights = flights[['DISTANCE', 'ELAPSED_TIME']].dropna()
Distance = flights['DISTANCE']
ElapsedTime = flights['ELAPSED_TIME']
# Distance[Distance != 0]= float(nan)
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
    output = []
    numberOfIterations = 0
    # for i in np.arange(0, iterations):
    #     h = inputMatrix.dot(weightVector)
    #     activationOutput = h
    #     print(weightVector)
    #     print(learningRate*np.dot(np.transpose(inputMatrix),activationOutput-targetOuput))
    #     weightVector -= learningRate*np.dot(np.transpose(inputMatrix),activationOutput-targetOuput)
    #     print(weightVector)
    #     # error = inputMatrix.T*((activationOutput-targetOuput)).T
    #     # weightDelta = error.sum(axis=1)
    #     # weightVector = weightVector - learningRate*weightDelta
    #     weightHistory = np.c_[weightHistory, weightVector]
    #     numberOfIterations +=1

        # if numberOfIterations == 5:
        #     break
        # if sum(abs(np.subtract(weightHistory[0:,weightHistory.shape[1]-2], weightVector))) < 0.1:
        #     break

    for iter in np.arange(0,iterations):
        error = 0
        for i in np.arange(0,inputMatrix.shape[0]):
            instance = inputMatrix[i]
            h = instance.dot(weightVector)
            output = 0 if(h < 0.5) else 1
            # error += 1 if(output != targetOuput[i]) else 0

            weightVector -= learningRate*(output-targetOuput[i])*instance

            weightHistory = np.c_[weightHistory, weightVector]
            # errorHistory = np.r_[errorHistory, error]

    # return weightVector,inputMatrix,weightHistory,errorHistory
    return weightVector,inputMatrix,weightHistory,numberOfIterations


######################### Training ###########################

xTrain,xTest,yTrain,yTest = trainingTest(Distance, ElapsedTime,0.8)
weightVector, X,weightHistory,numberOfIterations = preceptron(xTrain,yTrain,0.001,20)
######################### Testing ###########################

######################### Plotting ###########################
output = X.dot(weightVector)

fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()
print(output)
ax.scatter(Distance, ElapsedTime, marker='x')
ax.plot(xTrain,output,'r')

fig1, ax1 = plt.subplots()

# print(weightHistory[1,0:].shape,np.arange(1,weightHistory.shape[1]+1))
# ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[1,0:], color='b')
# ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[2,0:], color='r')

plt.show()