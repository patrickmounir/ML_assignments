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

param_range_HL =  [x for x in itertools.product((300,400,500,600,700),repeat=1)] + [x for x in itertools.product((300,400,500,600,700),repeat=2)]

train_scores,valid_scores = validation_curve(MLPClassifier(random_state=101),X_train,Y_train,'hidden_layer_sizes',param_range_HL,cv=3,verbose=True,n_jobs=-1)

np.savetxt('MLPTrainScore.txt', train_scores)
np.savetxt('MLPValidScore.txt', valid_scores)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with MLP")
plt.xlabel("Number of layers")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(range(1,len(param_range_HL)+1), train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(range(1,len(param_range_HL)+1), train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
#plt.semilogx(param_range_HL, test_scores_mean, label="Cross-validation score",
#             color="navy", lw=lw)
#plt.fill_between(param_range_HL, test_scores_mean - test_scores_std,
#                test_scores_mean + test_scores_std, alpha=0.2,
#                color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
# param_range_alpha = [0.00001,0.001,0.1,1,10]
# train_scores,valid_scores = validation_curve(MLPClassifier(random_state=101,hidden_layer_sizes=result you got from A),data,targets,'alpha',param_range_alpha,cv=3,verbose=True,n_jobs=-1)
# np.savetxt('MLPTrainScoreAlpha.txt', train_scores)
# np.savetxt('MLPValidScoreAlpha.txt', valid_scores)
param_range_gamma = [1,0.1,0.01,0.001,0.0001]
train_scores,valid_scores = validation_curve(SVC(random_state=101),X_train,Y_train,'gamma',param_range_gamma,cv=3,verbose=True,n_jobs=-1)
np.savetxt('SVCTrainScoreGamma.txt', train_scores)
np.savetxt('SVCValidScoreGamma.txt', valid_scores)
#  param_range_C = [0.1,1,10,100,1000]
# train_scores,valid_scores = validation_curve(SVC(random_state=101,gamma=result from A),data,targets,'C',param_range_C,cv=3,verbose=True,n_jobs=-1)
# np.savetxt('SVCTrainScoreC.txt', train_scores)
# np.savetxt('SVCValidScoreC.txt', valid_scores)
