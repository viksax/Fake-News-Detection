"""
    generate the k-fold cross-validation data
    """

import numpy as np
from scipy import stats
import math


def genData(k=10, file="./data.csv", writeToFile=True):
    
    train = []
    test = []
    if k == 1:
        raise ValueError('k should be higher than 1')
        return train, test
    
    inf = open(file)
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    
    k_length = int(math.floor((1.0 / k) * data.shape[0]))
    
    test.append(data[0:k_length])
    train.append(data[k_length:])

   
    for i in range(2, k):
        test.append(data[(i - 1) * k_length:i * k_length])
        train.append(np.append(data[:(i - 1) * k_length], data[i * k_length:], axis=0))
        pass

   
    test.append(data[(k - 1) * k_length:])
    train.append(data[:(k - 1) * k_length])

    if writeToFile:
        for i in range(1, k + 1):
            print("writing the file for " + str(i) + "-th cross validation")
            try:
                np.savetxt(str(i)+"train.csv", train[i-1], delimiter=",")
                np.savetxt(str(i)+"test.csv", test[i-1], delimiter=",")
            except:
                print("exception when writing the file for " + str(i) + "-th cross-validation")

            pass

    return train, test


if __name__ == "__main__":
    train, test = genData(10)
