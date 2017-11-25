import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.svm import SVC


X_train = np.loadtxt("X_train.txt", dtype=float)
scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar = scalar.fit(X_train)
X_train = scalar.transform(X_train)

X_test = np.loadtxt("X_test.txt", dtype=float)

Y_train = np.loadtxt("y_train.txt", dtype=int)

Y_test = np.loadtxt("y_test.txt", dtype=int)

# classifier = MLPClassifier(solver='adam', random_state=101,
#     hidden_layer_sizes=(500,),max_iter=10000,alpha=0.001,activation='tanh',batch_size=64,
#     learning_rate='constant',learning_rate_init=0.0001, validation_fraction=0.1,early_stopping=True,shuffle=True, verbose=True)
# classifier.fit(X_train, Y_train)
# print(Y_test)
# predictions = classifier.predict(X_test)
# print(predictions)
# print(precision_score(Y_test, predictions, average='micro'))

svc = SVC(C=100, gamma=0.001, random_state=101,  verbose=True)

svc.fit(X_train, Y_train)
print(Y_test)
predictions = svc.predict(X_test)
print(predictions)
print(precision_score(Y_test, predictions, average='micro'))
