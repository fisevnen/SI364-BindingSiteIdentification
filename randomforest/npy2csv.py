import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# set time
start=time.time()
# npfile = np.load(r'part1_add0_to600_train.npy')
npfile = np.load(r'total_feature.npy')
print(npfile.shape)
# np.set_printoptions(threshold=np.inf)
new_data = npfile.reshape(13178, 27000)
# print(np.shape(new_data))
# label = np.load(r'part1_add0_to600_label_train.npy')
label = np.load(r'total_label.npy')
# print(np.shape(label))



Xtrain, Xtest, Ytrain, Ytest = train_test_split(new_data, label, random_state=0)
print(f"Xtrain, Xtest, Ytrain, Ytest: ",len(Xtrain),'\n', len(Xtest), '\n', len(Ytrain),'\n', len(Ytest), '\n',)
forest = RandomForestClassifier()#100
forest.fit(Xtrain, Ytrain)


 


end=time.time()
print('running time={} s'.format(end-start))
print('n_estimators:{}, balanced:{}, random_state:{}, n_jobs:{}'.format(forest.n_estimators, forest.class_weight,forest.random_state,forest.n_jobs))
print("Accuracy on training set:{:.3f}".format(forest.score(Xtrain,Ytrain)))
print("Accuracy on test set:{:.3f}".format(forest.score(Xtest,Ytest)))
