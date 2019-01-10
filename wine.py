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
#以下三行是顯示其他屬性對quality的影響係數，並正序排列
z1 = wine.corr()
sorted(abs(z1['quality']))
z1_key = wine.keys()


#原始参数 ----------------------------------------------------------------------

wine_X= wine.drop(['quality'],axis=1)
wine_y= wine.quality

#wine_X = wine.iloc[:, 0:10]


wine_X=preprocessing.scale(wine_X)

X_train, X_test, y_train, y_test = train_test_split(
    wine_X, wine_y, test_size=0.3)#程序每运行一次，都会得到不同的准确率，无法调参。这个时候就是因为没有加random_state

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Original Training Score: ',knn.score(X_train, y_train))
print('Original Testing Score:',knn.score(X_test, y_test))

#降低方差
scores = cross_val_score(knn, wine_X, wine_y, cv=5, scoring='accuracy')#使用K折交叉验证模块
print(scores.mean())




#建立测试参数集(一般来说准确率(accuracy)会用于判断分类(Classification)模型的好坏)
k_range = range(1, 31)
k_scores = []

#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,wine_X, wine_y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    
#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


#最佳组合-------------------------------------------------------------------------
print('特征筛选及最佳超参数','-'*60)
wine_X1= wine.drop(['quality', 'pH', 'residual sugar', 'free sulfur dioxide'],axis=1)
wine_X1=preprocessing.scale(wine_X1)

X_train, X_test, y_train, y_test = train_test_split(
    wine_X1, wine_y, test_size=0.3)#程序每运行一次，都会得到不同的准确率，无法调参。这个时候就是因为没有加random_state


#knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=28)
knn.fit(X_train, y_train)
print('Final Training Score: ',knn.score(X_train, y_train))
print('Final Testing Score:',knn.score(X_test, y_test))
scores = cross_val_score(knn, wine_X1, wine_y, cv=5, scoring='accuracy')#使用K折交叉验证模块
print(scores.mean())


##一般来说平均方差(Mean squared error)会用于判断回归(Regression)模型的好坏
#k_range = range(1, 5)
#k_scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    loss = -cross_val_score(knn, wine_X, wine_y, cv=10, scoring='neg_mean_squared_error')
#    k_scores.append(loss.mean())
#
#plt.plot(k_range, k_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross-Validated MSE')
#plt.show()

#https://www.cnblogs.com/wf-ml/p/9615398.html







