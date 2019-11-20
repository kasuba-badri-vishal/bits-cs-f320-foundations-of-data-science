from Optimizers import StdGradDesc
from Optimizers import NormEqns
from Optimizers import RegGradDesc
from PolynomialModel.ModelStats import *


def transform_dataset(data, degree=2):
    """modifies the dataset to support polyonomial regression by adding features
    """

    degree = int(degree)
    if len(data.columns) == 3:
        data.columns = ['x1', 'x2', 'y']
    elif len(data.columns) == 4:
        data.columns = ['Const', 'x1', 'x2', 'y']
    if degree==1: return data

    for d in range(2,degree+1):
        for x in range(d+1):
            data[ 'x1^'+str(x)+'x2^'+str(d-x) ] = (data.loc[:, 'x1']**x)*(data.loc[:, 'x2']**(d-x))
    
    return data


def fit(x_train, y_train, alpha, epsilion, method="GD"):
    """fits the linear model to the given data
    """

    module = StdGradDesc
    if method == "NE": module = NormEqns

    w = module.run(x_train, y_train, function=f, error=error, alpha=alpha, epsilion=epsilion)
    return w


def reg_fit(train, alpha, epsilion, method="L2GD", lamdas=[], degree=0):
    """
    """
    module = RegGradDesc
    return module.run(train, function=f, error=error, alpha=alpha, epsilion=epsilion, reg_type=method, lamdas=lamdas, degree=degree)