import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



######################### Reading the file m_cerditcard.csv ###########################

creditcardTransactions = pd.read_csv('m_creditcard.csv')
features = creditcardTransactions[['V2', 'V11']]
isFraud = creditcardTransactions['Class']
######################### Normalization implementation ###########################
def normalize(X):
    normX = (X-X.mean())/X.std()
    return normX

######################### Mean Square Error Implementation ###########################

def calcError(pred, label):
    return np.sqrt(((pred - label) ** 2).mean())

######################### preceptron implementation ###########################

def preceptron(X, Y, learningRate):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)

    inputMatrix = np.c_[np.ones(inputMatrix.shape[0]), inputMatrix]

    weightVector = np.random.rand(inputMatrix.shape[1])
    weightHistory = weightVector
    errorHistory = []
    numberOfIterations = 0
    while True:
        h = np.dot(inputMatrix,weightVector)
        h[np.where(h >= 0.5)] = 1
        h[np.where(h < 0.5)] = 0
        activationOutput = h
        weightVector -= learningRate*np.dot(np.transpose(inputMatrix),activationOutput-targetOuput)
        error = calcError(activationOutput, targetOuput)
        weightHistory = np.c_[weightHistory, weightVector]
        numberOfIterations += 1
        if len(errorHistory) > 1 and abs(errorHistory[len(errorHistory)-1]-error) < 0.0000001:
            break
        errorHistory = np.r_[errorHistory, error]

    return weightVector,weightHistory,numberOfIterations


######################### Training ###########################

weightVector,weightHistory,numberOfIterations = preceptron(features,isFraud,0.2)
######################### Calculating the output ###########################
xAxis = np.array(features[['V11']])
xAxis = np.linspace(xAxis.min()+2,xAxis.max()+2)
y_plot = (-1/weightVector[1])*(weightVector[2]*xAxis+weightVector[0])
######################### Plotting ###########################

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(features[['V11']][isFraud==0],features[['V2']][isFraud==0], color='b', label="Genuine")
ax.scatter(features[['V11']][isFraud==1], features[['V2']][isFraud==1], color='r', label="Fraud")
ax.plot(xAxis,y_plot,c='g')
ax.legend()
ax.set_xlabel('V11')
ax.set_ylabel('V2')

fig1, ax1 = plt.subplots()
print(numberOfIterations)
ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[1,0:], color='b', label='Weight 1')
ax1.scatter(np.arange(1,numberOfIterations+2),weightHistory[2,0:], color='r', label='Weight 2')

ax1.set_xlabel('No. of Iterations')
ax1.set_ylabel('Weights')

plt.show()