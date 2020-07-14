# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:11:30 2020

@author: Acioli
"""


import sys
sys.path.append("../tools/")
from projeto import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

X_train, X_test, y_train, y_test = preprocess()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

preds = clf.predict(X_test)

acc_scor = accuracy_score(y_test,preds)
prec_scor = precision_score(y_test,preds)
f1_scor = f1_score(y_test,preds)
rec_scor = recall_score(y_test,preds)

print("Accuracy: {}\n".format(acc_scor) + 
      "Precision: {}\n".format(prec_scor) + 
      "Recall: {}\n".format(rec_scor) + 
      "F1: {}\n".format(f1_scor))