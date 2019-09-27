import theano
from   keras  import backend as K
import numpy                 as np

g_metric = K.variable(np.array([1]))

mtx    = {}
mtx[1] = [[1,2,3],[4,5,6],[7,8,9]]
mtx[2] = [[11,22],[33,44]]
mtx[3] = [[1,2],[3,4]]
mtx[4] = np.zeros((6,6))
mtx[5] = np.zeros((3,6))
mtx[6] = np.zeros((6,3))

mtx[7] = [
          [[13,52],[23,94]],
          [[45,64],[67,82]],
                        ]

vs = { i: K.variable( np.array(mtx[i]) ) for i in mtx}


#vs0 = K.stack((vs[3],vs[2]), axis=0)
"""
vs1 = K.concatenate((vs[4],vs[5]), axis=0)
vs2 = K.concatenate((vs[6],vs[1]), axis=0)
vs3 = K.concatenate((vs1,vs2), axis=1)
print K.eval( vs1 )
print K.eval( vs2 )
print K.eval( vs3 )
"""

magic1     = K.repeat_elements(K.eye(2), 2, 1)
magic2     = K.tile(K.eye(2), [1, 2])
magic_diff = magic1-magic2

print K.eval(magic_diff)

x  = vs[7]
d2 = K.reshape(K.expand_dims(K.dot(x, magic_diff), -1), (x.shape[0], x.shape[1], x.shape[2], x.shape[2]))

d2 = K.pow(d2, 2)


m_d2 = theano.tensor.tensordot(d2, g_metric, axes=0)

#x    = theano.tensor.batched_dot(vs[2],vs[2])



xx = theano.tensor.matrix('xx')
yy = theano.tensor.matrix('yy')

def lf(a,b):    return a+b



ff = theano.function([xx,yy], lf)


print K.eval(d2)
print K.eval(m_d2)

#print K.eval(x)
print K.eval(ff(vs[2],vs[2]))
















#a1 = [[1,2,3],[4,5,6],[7,8,9]]
#v1 = K.variable( np.array(a1) )
#a2 = [[11,22],[33,44]]
#v2 = K.variable( np.array(a2) )
#print K.eval(v1)
