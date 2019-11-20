from Optimizers import StdGradDesc
from Optimizers import StocasticGradDesc
from Optimizers import NormEqns
from Optimizers import RegGradDesc
from LinearModel.LinearModelStats import f, loss, error, test, r2_error


def fit(x_train, y_train, alpha, epsilion, method="GD"):
    """fits the linear model to the given data
    """

    module = StdGradDesc
    if method=="NE": module=NormEqns
    elif method=="GD": module=StdGradDesc
    elif method=="SGD": module=StocasticGradDesc

    w = module.run(x_train, y_train, function=f, error=loss, alpha=alpha, epsilion=epsilion)
    return w


def reg_fit(train, alpha, epsilion, method="L2GD", lamdas=[]):
    """
    """
    module = RegGradDesc
    return module.run(train, function=f, error=error, alpha=alpha, epsilion=epsilion, reg_type=method, lamdas=lamdas)