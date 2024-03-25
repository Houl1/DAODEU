from __future__ import division, print_function
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,average_precision_score,f1_score,mean_squared_error,mean_absolute_error
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from math import e

def get_score_t(real_activity,pre_activity):
    # print(pre_activity)
    # pre_activity_new = np.around(pre_activity) 
    # accuracy_score_t = accuracy_score(real_activity, pre_activity)
    # recall_score_t = recall_score(real_activity, pre_activity,labels=None, pos_label=1, average="micro", sample_weight=None)
    x = pre_activity
    lable =  np.squeeze(real_activity.astype(int))
    X_train, X_validate_test, y_train, y_validate_test = train_test_split(x, 
            lable, test_size=0.8)
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate_test, 
            y_validate_test, test_size=0.2/0.6)
    logistic = linear_model.LogisticRegression(solver="liblinear",max_iter=500)
    logistic.fit(X_train, y_train)
    test_predict = logistic.predict_proba(X_test)[:, 1]
    val_predict = logistic.predict_proba(X_validate)
    mean_squared_error_t = mean_squared_error(y_test, test_predict)
    mean_absolute_error_t = mean_absolute_error(y_test, test_predict)
    # print(pre_activity_new)
    # f1_score_t  = f1_score(real_activity, pre_activity,labels=None, pos_label=1, average="micro", sample_weight=None)
    return mean_absolute_error_t,mean_squared_error_t
