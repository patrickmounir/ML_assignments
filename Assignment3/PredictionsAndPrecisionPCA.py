import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA


X_train = np.loadtxt("X_train.txt", dtype=float)
scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar = scalar.fit(X_train)
X_train = scalar.transform(X_train)

X_test = np.loadtxt("X_test.txt", dtype=float)
X_test = scalar.transform(X_test)
Y_train = np.loadtxt("y_train.txt", dtype=int)

Y_test = np.loadtxt("y_test.txt", dtype=int)

n = [5,50,200,500]
pca = PCA(n_components=n[0])
pca.fit(X_train)
X_pca = pca.transform(X_train)
X_pcaTest = pca.transform(X_test)
classifier = MLPClassifier(solver='adam', random_state=101,
    hidden_layer_sizes=(500,),max_iter=10000,alpha=0.001,activation='tanh',batch_size=1,
    learning_rate='adaptive',learning_rate_init=0.0001, validation_fraction=0.1,early_stopping=True,shuffle=False, verbose=True)
classifier.fit(X_pca, Y_train)
print(Y_test)
predictions = classifier.predict(X_pcaTest)
print(predictions)
print(precision_score(Y_test, predictions, average='micro'))

svc = SVC(C=1000, gamma=0.001, random_state=101,  verbose=True)

svc.fit(X_pca, Y_train)
print(Y_test)
predictions = svc.predict(X_pcaTest)
print(predictions)
print(precision_score(Y_test, predictions, average='micro'))
