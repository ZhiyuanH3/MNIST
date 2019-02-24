from keras .datasets import mnist
import numpy as np
import pickle

(X_train, y_train), (X_test, y_test) = mnist.load_data()

npxt = np.array(X_test)
print np.array(X_test).shape
print npxt[3]

dataDict = {}
dataDict['X_train'] = X_train
dataDict['X_test'] = X_test
dataDict['y_test'] = y_test
dataDict['y_train'] = y_train

fileName = 'mnist.data'
fw = open(fileName,'wb')
pickle.dump(dataDict,fw)

fw.close()

