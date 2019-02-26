from   keras .datasets import mnist
## Load data:
(X_train, y_train), (X_test, y_test) = mnist.load_data()
n_event = 1000
X_train = X_train[:n_event]
y_train = y_train[:n_event]

"""
## Back up data:
import pickle
dataDict            = {}
dataDict['X_train'] = X_train
dataDict['X_test']  = X_test
dataDict['y_test']  = y_test
dataDict['y_train'] = y_train
with open('mnist.data', 'wb') as fw:    pickle.dump(dataDict,fw)
"""

import sys
sys.path.append('./')
import numpy  as np
import pandas as pd

from keras                      import backend as K
from keras                      import Sequential
from keras.layers               import Flatten, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D#, AveragePooling2D, ZeroPadding2D,
from keras.engine.topology      import Layer
from keras                      import initializers
from keras.utils                import np_utils
import                                 theano  as T
from sklearn.metrics            import roc_auc_score#, log_loss, classification_report, confusion_matrix, roc_curve
"""
from keras.callbacks  import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adamax, Nadam, Adadelta, Adagrad, RMSprop
from sklearn.utils    import class_weight
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization        import BatchNormalization
from keras.layers.core                 import Reshape
from keras.models                      import model_from_yaml

from sklearn               import preprocessing
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.multiclass    import OutputCodeClassifier
from sklearn.multiclass    import OneVsRestClassifier
from sklearn.multiclass    import OneVsOneClassifier
from sklearn.preprocessing import normalize
from sklearn.svm           import LinearSVC
from sklearn.preprocessing import StandardScaler
"""
######################################################
## Custom layers:
######################################################
class myDense(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(myDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w', 
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='RandomNormal',#'he_normal',#'zeros',#'uniform',     #'glorot_uniform',
                                 trainable=True)
        
        self.b = self.add_weight(name='b', 
                                 shape=(self.output_dim,),
                                 initializer=initializers.get('zeros'),
                                 trainable=True)
        super(myDense, self).build(input_shape)  

    def call(self, x):    
        return K.dot(x, self.w) + self.b
        #return K.bias_add( K.dot(x, self.w), self.b )

    def compute_output_shape(self, input_shape):    return (input_shape[0], self.output_dim)



class myDense_1(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(myDense_1, self).__init__(**kwargs)

    def build(self, input_shape):
        ## Variance 1:
        """    
        self.w = self.add_weight(name='w', 
                                 shape=(input_shape[1]-0, self.output_dim),
                                 initializer='RandomNormal',#'he_normal',#'zeros',#'uniform',     #'glorot_uniform',
                                 trainable=True)
        """

        ## Variance 2: 
        #self.ww = K.variable( np.array([[0,0],[0,0]]) )
        #print 'weight data type: ', self.ww.dtype
         
        self.www  = K.variable( np.array([1,2]) )
        self.wwww = K.variable( np.array([4,4]) )
        
        ## Not working:
        #self.ww   = K.variable( K.stack((self.www, self.wwww),axis=0) )  
        
        ## What is this method for?
        #self.set_weights()

        self.trainable_weights = [self.www,self.wwww]#[self.www]
        
        self.b = self.add_weight(name='b', 
                                 shape=(self.output_dim,),
                                 initializer=initializers.get('zeros'),
                                 trainable=True)
        super(myDense_1, self).build(input_shape)  

    def call(self, x):
        ## Variance 1:
        #w_4 = self.w
        #return K.bias_add( K.dot(x, w_4), self.b )

        ## Variance 2:   
        #return K.bias_add( K.dot(x, self.ww), self.b ).astype('float32')
        
        ## Variance 3:   
        dp = K.dot(x, K.stack((self.www,self.wwww),axis=0))
        return K.bias_add( dp, self.b ).astype('float32')


    def compute_output_shape(self, input_shape):    return (input_shape[0], self.output_dim)

######################################################
######################################################










######################################
## Simple test:
######################################

## Show input shapes:
d_shape = X_test.shape
print d_shape
print y_test.shape
pix_dim = d_shape[1]

## Binary classification:
n_class             = 2 
print y_test
y_test[ y_test !=1] = 0
y_train[y_train!=1] = 0
print y_test

## Calculate weights:
W_train = np.array(y_train, dtype='float').copy()
W_test  = np.array(y_test , dtype='float').copy()
n_sgn_train   = (W_train==1).sum()
n_sgn_test    = (W_test==1).sum()
n_bkg_train   = (W_train==0).sum()
n_bkg_test    = (W_test==0).sum()
print 'n_sgn_train: ', n_sgn_train
print 'n_bkg_train: ', n_bkg_train
print 'n_sgn_test: ', n_sgn_test
print 'n_bkg_test: ', n_bkg_test
W_train[W_train==1] = 1./n_sgn_train
W_train[W_train==0] = 1./n_bkg_train
W_test[W_test==1]   = 1./n_sgn_test
W_test[W_test==0]   = 1./n_bkg_test


## One-hot:
Y_train = np_utils.to_categorical(y_train, num_classes=n_class) 
Y_test  = np_utils.to_categorical(y_test , num_classes=n_class)

## Reproducibility:
rd_seed   = 4
np.random.seed(seed=rd_seed)

vb        = 2#1
Optimizer = 'adam'
loss_func = 'categorical_crossentropy'
Metric    = ['accuracy']

n_nodes   = 2
n_epochs  = 10#4#20#10#2#3
batch_s   = 32

"""
ST        = Sequential()
ST.add( Flatten(input_shape=(pix_dim,pix_dim,)) )
ST.add( Dense(n_nodes, activation='relu') )
ST.add( Dense(n_class, activation='softmax') )


ST.compile(
            optimizer        = Optimizer, 
            loss             = loss_func, 
            metrics          = Metric,
            weighted_metrics = ['accuracy']
          )

ST.fit(
       x             = X_train, 
       y             = Y_train, 
       sample_weight = W_train,
       epochs        = n_epochs, 
       verbose       = vb, 
       shuffle       = True, 
       batch_size    = batch_s
      )

ST.summary()
"""

"""
##############################
np.random.seed(seed=rd_seed)
ST1        = Sequential()
ST1.add( Flatten(input_shape=(pix_dim,pix_dim,)) )
ST1.add( Dense(2, activation='relu') )
ST1.add( Dense(n_class, activation='softmax') )
ST1.compile(
            optimizer        = Optimizer, 
            loss             = loss_func, 
            metrics          = Metric,
            weighted_metrics = ['accuracy']
          )
ST1.fit(
       x             = X_train, 
       y             = Y_train, 
       sample_weight = W_train,
       epochs        = n_epochs, 
       verbose       = vb, 
       shuffle       = True, 
       batch_size    = batch_s
      )
ST1.summary()
##############################
"""

## Test custom layer###########################
np.random.seed(seed=rd_seed)
ST2        = Sequential()
ST2.add( Flatten(input_shape=(pix_dim,pix_dim,)) )

#ST2.add( myDense(3) )
#ST2.add( Activation('relu') )
ST2.add( myDense(n_nodes) )
ST2.add( Activation('relu') )

#ST2.add( myDense_1(n_nodes) )
#ST2.add( Activation('relu') )

ST2.add( myDense_1(n_class) )
ST2.add( Activation('softmax') )

#ST2.add( Dense(n_class, activation='softmax') )
ST2.compile(
            optimizer        = Optimizer, 
            loss             = loss_func, 
            metrics          = Metric,
            weighted_metrics = ['accuracy']
          )
ST2.fit(
       x             = X_train, 
       y             = Y_train, 
       sample_weight = W_train,
       epochs        = n_epochs, 
       verbose       = vb, 
       shuffle       = True, 
       batch_size    = batch_s
      )
ST2.summary()
## Test custom layer###########################










STs = [ST2]

for STi in STs:
    ## Show learned weights:
    ii = 1
    for layer in STi.layers:
        g = layer.get_config()
        h = layer.get_weights()
        if ii == 4:
            print '------------------------------------ Layer ', str(ii)
            print g
            print h
        ii += 1  


    ## Accuracy:
    acc = STi.evaluate(
                      x             = X_test, 
                      y             = Y_test,
                      sample_weight = W_test
                    )
    print 'Acc: ', acc[1]


    ## AUC:
    y_test_prd = STi.predict(
                            x          = X_test, 
                            batch_size = batch_s,
                            verbose    = vb
                          )

    ## Pick signal probabilities:
    #print y_test_prd.shape
    y_test_prd = y_test_prd[:,1]
    #print y_test_prd.shape

    AUC = roc_auc_score(
                        y_test, 
                        y_test_prd, 
                        sample_weight = W_test
                      )
    print('AUC: {0}'.format(AUC))















