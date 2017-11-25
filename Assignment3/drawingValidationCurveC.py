import numpy as np
import itertools
import matplotlib.pyplot as plt

param_range = [0.1,1,10,100,1000]
train_scores = np.loadtxt("SVCTrainScoreC.txt", dtype=float)

valid_scores = np.loadtxt("SVCValidScoreC.txt", dtype=float)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)

print('Parameters: ')
print(param_range[4])
print('Mean of validation: ')
print(test_scores_mean[4])
print('Mean of training: ')
print(train_scores_mean[4])

plt.title("Validation Curve with SVC")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.scatter(param_range[4], train_scores_mean[4], color="darkorange")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
            color="navy", lw=lw)
plt.scatter(param_range[4], test_scores_mean[4],color="navy",label="Opitmal Point")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
               test_scores_mean + test_scores_std, alpha=0.2,
               color="navy", lw=lw)
plt.legend(loc="best")
plt.show()