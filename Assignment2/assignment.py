from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('timesereis_8_2.csv')
features = data[['0', '1','2', '3','4', '5','6', '7']]
target=data[['8','9']]

XTrain, XTest, YTrain, YTest = train_test_split(features, target, test_size=0.2,random_state=4)

#####################MLP######################################
def MLP(X, Y, learningRate,layers):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)
    Lambda = 1e-05
    BS = 128
    reg = MLPRegressor(solver='sgd',alpha=Lambda,batch_size=BS,
        hidden_layer_sizes=layers,max_iter=10000,
        learning_rate='constant',learning_rate_init=learningRate)
  
    return reg.fit(inputMatrix,targetOuput)

############################Main#############################
#Configuration 2 hidden layers where the first is 10 nodes and the second is 8 nodes
reg1 = MLP(XTrain,YTrain,0.001,(10,8))
#Configuration 1 hidden layer with 100 nodes
reg2 = MLP(XTrain,YTrain, 0.001, (100,))
#Configuration 2 hidden nodes with 50 foe the first and 10 for the second
reg3 = MLP(XTrain,YTrain,0.001,(90,10))

hidden_layers = np.array([[10,8],[100],[50,10]])

reg1Prediction = reg1.predict(XTest)
print("Mean absolute error on the testing data for 1st configuration of hidden layers:")
print(metrics.mean_absolute_error(YTest,reg1Prediction))

reg2Prediction = reg2.predict(XTest)
print("Mean absolute error on the testing data for 2nd configuration of hidden layers:")
print(metrics.mean_absolute_error(YTest,reg2Prediction))


reg3Prediction = reg3.predict(XTest)


print("Mean absolute error on the testing data for 3rd configuration of hidden layers:")
print(metrics.mean_absolute_error(YTest,reg3Prediction))



#Configuration learning rate 0.001
reg1Learning = MLP(XTrain,YTrain,0.001,(10,8))
#Configuration leaarning rate 0.00001
reg2Learning = MLP(XTrain,YTrain, 0.00001, (10,8))



reg1LearningPrediction = reg1Learning.predict(XTest)
print("Mean absolute error on the testing data for 1st configuration of learningRate:")
print(metrics.mean_absolute_error(YTest,reg1LearningPrediction))
reg2LearningPrediction = reg2Learning.predict(XTest)
print("Mean absolute error on the testing data for 2nd configuration of learningRate:")
print(metrics.mean_absolute_error(YTest,reg2LearningPrediction))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes.ravel()[0].plot(np.arange(0, reg1.n_iter_), reg1.loss_curve_)

axes.ravel()[1].plot(np.arange(0, reg2.n_iter_), reg2.loss_curve_)

axes.ravel()[2].plot(np.arange(0, reg3.n_iter_), reg3.loss_curve_)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 10))
axes2.ravel()[0].plot(np.arange(0, reg1Learning.n_iter_), reg1Learning.loss_curve_)

axes2.ravel()[1].plot(np.arange(0, reg2Learning.n_iter_), reg2Learning.loss_curve_)

plt.show()