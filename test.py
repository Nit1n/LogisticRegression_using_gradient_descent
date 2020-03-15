from binary_classifier1 import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


log_clf = LogisticRegression()
iris = load_iris()
x = iris.data
y = (iris.target ==2)

x_train ,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)
log_clf.fit(x_train,y_train)
y_pred = log_clf.predict(x_test)
pre_score = precision_score(y_test,y_pred)
print('theta :' , log_clf.theta)
print('predicted values',y_pred)
print('precision_score' , pre_score)