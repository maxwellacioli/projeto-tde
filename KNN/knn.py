import sys
sys.path.append("../tools/")
from projeto import preprocess
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test, data, target = preprocess()

neigh = KNeighborsClassifier(n_neighbors=7)
neigh = neigh.fit(X_train, y_train)

pred = neigh.predict(X_test)

acc_scor = accuracy_score(y_test, pred)
f1_scor = f1_score(y_test, pred)
prec_scor = precision_score(y_test, pred)
rec_scor = recall_score(y_test, pred)

print("Accuracy: {}".format(acc_scor) + "     Precision: {}".format(prec_scor) + "     Recall: {}".format(rec_scor) + "     F1: {}".format(f1_scor))