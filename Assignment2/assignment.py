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

XTrain, XTest, YTrain, YTest = train_test_split(features, target, test_size=0.2,random_state=0)

#####################MLP######################################
def MLP(X, Y, learningRate,layers, early_stop=True, Lambda=1e-05,BS=128):
    inputMatrix = np.array(X)
    targetOuput = np.array(Y)
    Lambda = 1e-05
    BS = 128
    reg = MLPRegressor(solver='sgd',
        hidden_layer_sizes=layers,max_iter=10000,
        learning_rate='constant',learning_rate_init=learningRate, validation_fraction=0.1,early_stopping=early_stop,shuffle=False)
  
    return reg.fit(inputMatrix,targetOuput)

############################Main#############################
#Configuration 2 hidden layers where the first is 10 nodes and the second is 8 nodes
reg1 = MLP(XTrain,YTrain,0.001,(10,8,20,10))
#Configuration 1 hidden layer with 100 nodes
reg2 = MLP(XTrain,YTrain, 0.001, (10,8))
#Configuration 2 hidden nodes with 50 foe the first and 10 for the second
reg3 = MLP(XTrain,YTrain,0.001,(10,8,20,10,5))



#Configuration learning rate 0.001
reg1Learning = MLP(XTrain,YTrain,0.001,(10,8))
#Configuration leaarning rate 0.00001
reg2Learning = MLP(XTrain,YTrain, 0.00001, (10,8))



reg1Prediction = reg1.predict(XTest)
print("Mean absolute error on the testing data for 1st configuration of hidden layers:")
print(metrics.mean_absolute_error(YTest,reg1Prediction))

reg2Prediction = reg2.predict(XTest)
print("Mean absolute error on the testing data for 2nd configuration of hidden layers:")
print(metrics.mean_absolute_error(YTest,reg2Prediction))


reg3Prediction = reg3.predict(XTest)


print("Mean absolute error on the testing data for 3rd configuration of hidden layers:")
print(metrics.mean_absolute_error(YTest,reg3Prediction))



print("Validation loss for 1st configuration of hidden layers:")
print(reg1.best_validation_score_)
print(reg1.loss_)


print("Validation loss for 2nd configuration of hidden layers:")
print(reg2.best_validation_score_)
print(reg2.loss_)



print("Validation loss for 3rd configuration of hidden layers:")
print(reg3.best_validation_score_)
print(reg3.loss_)

Lambda = 1e-05
reg2Regularized = MLPRegressor(solver='sgd', alpha=Lambda,
        hidden_layer_sizes=(10,8),max_iter=10000,
        learning_rate='constant',learning_rate_init=0.001, validation_fraction=0.1,early_stopping=True,shuffle=False)
reg2Regularized.fit(XTrain,YTrain)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes.ravel()[0].plot(np.arange(0, reg1.n_iter_), reg1.loss_curve_, color="g", label="Training Loss")
axes.ravel()[0].plot(np.arange(0, reg1.n_iter_), reg1.validation_scores_, color='r', label="Validation loss")
axes.ravel()[0].set_title("Number of layer "+str(reg1.hidden_layer_sizes))
axes.ravel()[0].set_xlabel("Number of iterations")
axes.ravel()[0].set_ylabel("Error")
axes.ravel()[0].legend()


axes.ravel()[1].plot(np.arange(0, reg2.n_iter_), reg2.loss_curve_, color='g', label="Training Loss")
axes.ravel()[1].plot(np.arange(0, reg2.n_iter_), reg2.validation_scores_, color='r', label="Validation loss")
axes.ravel()[1].set_title("Number of layer "+str(reg2.hidden_layer_sizes))
axes.ravel()[1].set_xlabel("Number of iterations")
axes.ravel()[1].set_ylabel("Error")
axes.ravel()[1].legend()

axes.ravel()[2].plot(np.arange(0, reg3.n_iter_), reg3.loss_curve_, color='g', label="Training Loss")
axes.ravel()[2].plot(np.arange(0, reg3.n_iter_), reg3.validation_scores_, color='r', label="Validation loss")
axes.ravel()[2].set_title("Number of layer "+str(reg3.hidden_layer_sizes))
axes.ravel()[2].set_xlabel("Number of iterations")
axes.ravel()[2].set_ylabel("Error")
axes.ravel()[2].legend()

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 10))
axes2.ravel()[0].plot(np.arange(0, reg1Learning.n_iter_), reg1Learning.loss_curve_,  color='g', label="Training Loss")
# axes2.ravel()[0].plot(np.arange(0, reg1Learning.n_iter_), reg1Learning.validation_scores_, color='r' ,label="Validation loss")
axes2.ravel()[0].set_title("learning rate "+str(reg1Learning.learning_rate_init))
axes2.ravel()[0].set_xlabel("Number of iterations")
axes2.ravel()[0].set_ylabel("Training Error")


axes2.ravel()[1].plot(np.arange(0, reg2Learning.n_iter_), reg2Learning.loss_curve_,color='g', label="Training loss")
# axes2.ravel()[1].plot(np.arange(0, reg2Learning.n_iter_), reg2Learning.validation_scores_, color='r' ,label="Validation Loss")
axes2.ravel()[1].set_title("learning rate "+str(reg2Learning.learning_rate_init))
axes2.ravel()[1].set_xlabel("Number of iterations")
axes2.ravel()[1].set_ylabel("Training Error")

fig3, axes3 = plt.subplots(1, 2, figsize=(15, 10))
axes3.ravel()[0].plot(np.arange(0, reg2.n_iter_), reg2.loss_curve_,  color='g', label="Training Loss")
axes3.ravel()[0].plot(np.arange(0, reg2.n_iter_), reg2.validation_scores_, color='r' ,label="Validation loss")
axes3.ravel()[0].set_title("un-Regularized")
axes3.ravel()[0].set_xlabel("Number of iterations")
axes3.ravel()[0].set_ylabel("Error")


axes3.ravel()[1].plot(np.arange(0, reg2Regularized.n_iter_), reg2Regularized.loss_curve_,color='g', label="Training loss")
axes3.ravel()[1].plot(np.arange(0, reg2Regularized.n_iter_), reg2Regularized.validation_scores_, color='r' ,label="Validation Loss")
axes3.ravel()[1].set_title("Regularized")
axes3.ravel()[1].set_xlabel("Number of iterations")
axes3.ravel()[1].set_ylabel("Error")

plt.show()