from pyGPs.Core.mean import Mean
import numpy as np

class Quadratic(Mean):
    '''
    Linear mean function. self.hyp = alpha_list

    :param D: dimension of training data. Set if you want default alpha, which is 0.5 for each dimension.
    :alpha_list: scalar alpha for each dimension
    '''
    def __init__(self, D=None, alpha_list=None):
        if alpha_list is None:
            if D is None:
                self.hyp = [0.5]
            else:
                self.hyp = [0.5 for i in range(D)]
        else:
            self.hyp = alpha_list

    def getMean(self, x=None):
        n, D = x.shape
        X = np.hstack((x,pow(x,2)))
        c = np.array(self.hyp)
        c = np.reshape(c,(len(c),1))
        A = np.dot(X,c)
        return A

    def getDerMatrix(self, x=None, der=None):
        n, D = x.shape
        X = np.hstack((x, pow(x, 2)))
        c = np.array(self.hyp)
        c = np.reshape(c,(len(c),1))
        if isinstance(der, int) and der < D:     # compute derivative vector wrt meanparameters
            A = np.reshape(X[:,der], (len(X[:,der]),1) )
        else:
            A = np.zeros((n,1))
        return A