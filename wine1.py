import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing #标准化数据模块
from sklearn.cross_validation import cross_val_score # K折交叉验证模块
import matplotlib.pyplot as plt #可视化模块
from sklearn.learning_curve import validation_curve #validation_curve模块
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV

wine = pd.read_csv('winequality-red.csv',header=0)
wine_X= wine.drop(['quality'],axis=1)
wine_y= wine.quality
wine_X=preprocessing.scale(wine_X)

X_train, X_test, y_train, y_test = train_test_split(
    wine_X, wine_y, test_size=0.3)#程序每运行一次，都会得到不同的准确率，无法调参。这个时候就是因为没有加random_state
knn = KNeighborsClassifier()

k_range = list(range(1,3))
leaf_range = list(range(1,2))
weight_options = ['uniform','distance']
algorithm_options = ['auto','ball_tree','kd_tree','brute']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
gridKNN = GridSearchCV(knn,param_gridknn,cv=10,scoring='accuracy',verbose=1)
gridKNN.fit(X_train,y_train)
print('best score is:',str(gridKNN.best_score_))
print('best params are:',str(gridKNN.best_params_))



