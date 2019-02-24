import pickle

fileName = 'mnist.data'

fr = open(fileName,'rb')
mnist = pickle.load(fr)
print mnist['X_test']
print mnist['y_test']
