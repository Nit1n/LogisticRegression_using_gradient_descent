import numpy as np
from sklearn.preprocessing import LabelEncoder

class LogisticRegression :
    def __init__(self,max_iter =1000,alpha = 0.001):
        self.max_iter  = max_iter
        self.alpha  = alpha

    def hypothesis(self,theta,x):

        h = 1/(1+np.exp(-(x.dot(theta))))
        return h

    def _error(self,y,x,m,theta,j):
        Sum = 0
        for i in  range(m):
            Sum += (y[i] - self.hypothesis(theta ,x[i]))*x[i][j]
        return Sum

    def _gradient_descent(self,x,y):
        # check dimension of the input array
        dim = x.ndim
        if dim ==1  :
            raise TypeError('reshape array feature to (-1,1 or (1,-1)' )
        else :
            self.n_samples , self.n_features = x.shape
            self.theta = np.zeros((self.n_features,1))
            for epoches in range(self.max_iter) :
                new_theta = np.full_like(self.theta,0)
                for j in range(self.n_features) :
                    new_theta[j] = self.theta[j] + self.alpha*(self._error(y,x,self.n_samples,self.theta,j))
                self.theta = new_theta
    def fit(self,x,y):
        new_y = LabelEncoder().fit_transform(y)
        self._gradient_descent(x,new_y)
        return self

    def predict(self,x):
        if x.ndim ==1 :
            raise TypeError('reshape array feature to (-1,1) or (1,-1) ')
        else :
            res = []
            test_m,test_n = x.shape
            if test_n != self.n_features :
                raise TypeError('features size must be of shape({},{})'.format(self.n_samples,self.n_features))
            else :
                for i in range(test_m) :
                    hypo = self.hypothesis(self.theta,x[i])
                    if hypo >=0.5 :
                        res.append(1)
                    else :
                        res.append(0)

            return np.array(res)


