import numpy as np
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


X_train = np.loadtxt("X_train.txt", dtype=float)
scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar = scalar.fit(X_train)
X_train = scalar.transform(X_train)

X_test = np.loadtxt("X_test.txt", dtype=float)

Y_train = np.loadtxt("y_train.txt", dtype=int)

Y_test = np.loadtxt("y_test.txt", dtype=int)

if __name__ == '__main__':
    param_range_HL =  [x for x in itertools.product((300,400,500,600,700),repeat=1)] + [x for x in itertools.product((300,400,500,600,700),repeat=2)]
    train_scores,valid_scores = validation_curve(MLPClassifier(random_state=101),X_train,Y_train,'hidden_layer_sizes',param_range_HL,cv=3,verbose=True,n_jobs=-1)
    np.savetxt('MLPTrainScore.txt', train_scores)
    np.savetxt('MLPValidScore.txt', valid_scores)
    
    param_range_alpha = [0.00001,0.001,0.1,1,10]
    train_scores,valid_scores = validation_curve(MLPClassifier(random_state=101,hidden_layer_sizes=(500,)),X_train,Y_train,'alpha',param_range_alpha,cv=3,verbose=True,n_jobs=-1)
    np.savetxt('MLPTrainScoreAlpha.txt', train_scores)
    np.savetxt('MLPValidScoreAlpha.txt', valid_scores)

    param_range_gamma = [1,0.1,0.01,0.001,0.0001]
    train_scores,valid_scores = validation_curve(SVC(random_state=101),X_train,Y_train,'gamma',param_range_gamma,cv=3,verbose=True,n_jobs=-1)
    np.savetxt('SVCTrainScoreGamma.txt', train_scores)
    np.savetxt('SVCValidScoreGamma.txt', valid_scores)

    param_range_C = [0.1,1,10,100,1000]
    train_scores,valid_scores = validation_curve(SVC(random_state=101,gamma=0.001),X_train,Y_train,'C',param_range_C,cv=3,verbose=True,n_jobs=-1)
    np.savetxt('SVCTrainScoreC.txt', train_scores)
    np.savetxt('SVCValidScoreC.txt', valid_scores)
