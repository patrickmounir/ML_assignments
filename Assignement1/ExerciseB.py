import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################### Reading the file HA.csv ###########################

flights = pd.read_csv('HA.csv')
flights = flights[['DISTANCE', 'ELAPSED_TIME']].dropna()
Distance = flights['DISTANCE']
ElapsedTime = flights['ELAPSED_TIME']
mx = 0
my = 0

# Distance[Distance != 0]= float(nan)
######################### Divide Training and Testing Implementation ###########################

def trainingTest(X, Y, trainingPercentage):
    X = np.array(X)
    # normalize X
    global mx, my
    mx = np.max(X)
    X = X / mx
    # # normalize Y
    Y = np.array(Y)
    # my = np.max(Y)
    # Y = Y / my

    trainingLength = int(X.shape[0] * trainingPercentage)
    xTrain = np.zeros(trainingLength)
    yTrain = np.zeros(trainingLength)
    xTest = np.zeros(X.shape[0] - trainingLength)
    yTest = np.zeros(X.shape[0] - trainingLength)
    for i in np.arange(0, trainingLength):
        xTrain[i] = X[i]
        yTrain[i] = Y[i]
    for i in np.arange(trainingLength, X.shape[0]):
        xTest[i - trainingLength] = X[i]
        yTest[i - trainingLength] = Y[i]

    return xTrain, xTest, yTrain, yTest
######################### Data Normalization Implementation ###########################
def normalize(X):
    normX = (X-X.mean())/X.std()
    return normX
######################### Mean Square Error Implementation ###########################

def calcError(pred, label):
    return np.sqrt(((pred - label) ** 2).mean())

######################### Preceptron Implementation ###########################

def preceptron(X, Y, learningRate):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)

    inputMatrix = np.c_[np.ones(inputMatrix.shape[0]), inputMatrix]

    weightVector = np.random.rand(inputMatrix.shape[1])
    weightHistory = weightVector
    errorHistory = []
    numberOfIterations = 0
    while True:
        h = inputMatrix.dot(weightVector)
        activationOutput = h
        weightVector -= learningRate * np.dot(np.transpose(inputMatrix),activationOutput - targetOuput)
        error = calcError(activationOutput, targetOuput)
        weightHistory = np.c_[weightHistory, weightVector]
        numberOfIterations += 1
        print(abs(errorHistory[len(errorHistory)-1]-error))
        
        if len(errorHistory) > 1 and abs(errorHistory[len(errorHistory)-1]-error) < 0.0000001 :
            break
        errorHistory = np.r_[errorHistory, error]
    return weightVector, weightHistory, numberOfIterations


######################### Training ###########################
xTrain, xTest, yTrain, yTest = trainingTest(Distance, ElapsedTime, 0.8)
weightVector, weightHistory, numberOfIterations = preceptron(xTrain, yTrain, 0.000001)
######################### Testing ###########################
X = np.c_[np.ones(xTrain.shape[0]), xTrain]
output = X.dot(weightVector)
output = output
xTestBiased = np.c_[np.ones(xTest.shape[0]), xTest]
outputTest = xTestBiased.dot(weightVector)
print(calcError(outputTest, yTest))
######################### Plotting ###########################
fig, ax = plt.subplots()

ax.scatter(Distance, ElapsedTime, marker='x',)
ax.plot(xTrain*mx, output, 'r')
ax.set_xlabel('Distance')
ax.set_ylabel('Elapsed Time')


fig1, ax1 = plt.subplots()
ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[1,0:], color='b')
ax1.set_xlabel('No. of Iterations')
ax1.set_ylabel('Weight')


plt.show()