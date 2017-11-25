import numpy as np
import itertools
import matplotlib.pyplot as plt

param_range_HL =  [x for x in itertools.product((300,400,500,600,700),repeat=1)] + [x for x in itertools.product((300,400,500,600,700),repeat=2)]
train_scores = np.loadtxt("MLPTrainScore.txt", dtype=float)

valid_scores = np.loadtxt("MLPValidScore.txt", dtype=float)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)

print('Parameters: ')
print(param_range_HL[2])
print('Mean of validation: ')
print(test_scores_mean[2])
print('Mean of training: ')
print(train_scores_mean[2])

plt.title("Validation Curve with MLP")
plt.xlabel("Number of layers")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(range(len(param_range_HL)), train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.scatter(2, train_scores_mean[2], color="darkorange")
plt.fill_between(range(len(param_range_HL)), train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(range(len(param_range_HL)), test_scores_mean, label="Cross-validation score",
            color="navy", lw=lw)
plt.scatter(2, test_scores_mean[2],color="navy",label="Opitmal Point")
plt.fill_between(range(len(param_range_HL)), test_scores_mean - test_scores_std,
               test_scores_mean + test_scores_std, alpha=0.2,
               color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
