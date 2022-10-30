import numpy as np

def MSE(y_gt, y_pred):
    return np.mean(np.power(y_gt - y_pred, 2))

def MSE_prim(y_gt, y_pred):
    return 2* (y_pred - y_gt) / np.size(y_gt)    



def binary_cross_entropy(y_gt, y_pred):
    return -np.mean(y_gt * np.log(y_pred) + (1 - y_gt) * np.log(1 - y_pred))


def binary_cross_entropy_prim(y_gt, y_pred):
    return ((1-y_gt) / (1 - y_pred) - y_true / y_pred) / np.size(y_gt)