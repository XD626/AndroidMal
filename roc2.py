from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import  AdaBoostClassifier,GradientBoostingClassifier
from sklearn import svm

import tflearn
from tflearn.layers import input_data,dropout,fully_connected
from tflearn.data_utils import to_categorical,pad_sequences
from tflearn.layers.conv import conv_1d,global_max_pool
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_y():
    y=[]
    y1 = [1] * 600 # malware
    y2 = [0] * 564# normal
    y = y1 + y2
    return y
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
def do_metrics(y_test,y_pred):
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(y_test,y_pred))
# knn
def do_knn():
    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    print("knn:")
    print(np.mean(model_selection.cross_val_score(knn, x, y, n_jobs=-1, cv=10)))
    y_pred = knn.predict(x_test)
    do_metrics(y_test, y_pred)
    return y_test, y_pred
# RandomForest
def do_rfc():
    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print("RandomForest:")
    y_pred = rfc.predict(x_test)
    do_metrics(y_test, y_pred)
    print("十倍交叉验证")
    # print(np.mean(model_selection.cross_val_score(rfc, x, y, n_jobs=-1, cv=10)))
    #print(model_selection.cross_val_score(rfc, x, y, n_jobs=-1, cv=10))
    return y_test, y_pred

# naive_bayes
def do_gnb():
    data = pd.read_csv('data\\3_gram.csv')
    y = get_y()
    x = data.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print("naive_bayes:")
    y_pred = gnb.predict(x_test)
    do_metrics(y_test, y_pred)
    # print(np.mean(model_selection.cross_val_score(gnb, x, y, n_jobs=-1, cv=10)))
    return y_test, y_pred
# AdaBoost
def do_adb(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    adb = AdaBoostClassifier(base_estimator=None, algorithm="SAMME", n_estimators=600, learning_rate=0.7,
                             random_state=None)
    adb.fit(x_train, y_train)
    print("AdaBoost:")
    y_pred = adb.predict(x_test)
    do_metrics(y_test, y_pred)
    #print(np.mean(model_selection.cross_val_score(adb, x, y, n_jobs=-1, cv=10)))
    return y_test, y_pred

#svm
def do_svm(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    clf_svm = svm.SVC()
    clf_svm.fit(x_train, y_train)
    print("svm:")
    y_pred = svm.predict(x_test)
    do_metrics(y_test, y_pred)
    print(np.mean(model_selection.cross_val_score(clf_svm, x, y, n_jobs=-1, cv=10)))



#gbdt
def do_gbdt(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    clf_gbdt = GradientBoostingClassifier(random_state=10)
    print("gbdt:")
    clf_gbdt.fit(x_train,y_train)
    y_pred = clf_gbdt.predict(x_test)
    do_metrics(y_test, y_pred)
    #print(np.mean(model_selection.cross_val_score(clf_gbdt, x, y, n_jobs=-1, cv=10)))
    return y_test, y_pred



if __name__ == '__main__':
    y_test_knn, y_score_knn = do_knn()
    # Compute ROC curve and ROC area for each class
    fpr_knn, tpr_knn, threshold = roc_curve(y_test_knn, y_score_knn)  ###计算真正率和假正率
    roc_auc = auc(fpr_knn, tpr_knn)  ###计算auc的值
    plt.figure()
    lw = 2
    #bayes
    y_test_bayes,y_score_bayes = do_gnb()
    fpr_bayes, tpr_bayes, threshold = roc_curve(y_test_bayes, y_score_bayes)  ###计算真正率和假正率

    # forest
    y_test_forest, y_score_forest = do_rfc()
    fpr_forest, tpr_forest, threshold = roc_curve(y_test_forest, y_score_forest)  ###计算真正率和假正率


    plt.figure(figsize=(10, 10))

    plt.plot(fpr_knn, tpr_knn, color='darkorange',linestyle ='--',
             label="knn")  ###假正率为横坐标，真正率为纵坐标做曲线  lw=lw , label='ROC curve (area = %0.2f)' % roc_auc
    plt.plot(fpr_bayes, tpr_bayes, color='green',linestyle ='-.',
             label="gnb")
    plt.plot(fpr_forest, tpr_forest, color='black',linestyle =':',
             label="rfc")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('E:\\Machine Learning\\1')
    plt.show()

