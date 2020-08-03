# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:11:30 2020

@author: Acioli
"""


import sys
sys.path.append("../tools/")
from projeto import preprocess
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test, data, target = preprocess()

clf = RandomForestClassifier(criterion='entropy', max_depth= 7, 
                             class_weight='balanced')

acc_scor = np.mean(cross_val_score(clf, data, target, cv=10, scoring='accuracy'))
prec_scor = np.mean(cross_val_score(clf, data, target, cv=10, scoring='precision'))
f1_scor = np.mean(cross_val_score(clf, data, target, cv=10, scoring='f1'))
rec_scor = np.mean(cross_val_score(clf, data, target, cv=10, scoring='recall'))

print("Accuracy: {}\n".format(acc_scor) + 
      "Precision: {}\n".format(prec_scor) + 
      "Recall: {}\n".format(rec_scor) + 
      "F1: {}\n".format(f1_scor))