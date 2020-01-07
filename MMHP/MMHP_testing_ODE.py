# Testing functions in Learning_MLE_ODE.py

import numpy.matlib
from Learning_MLE_ODE import *
from Simulation_Branch_HP import *
from classes import *
from classes_skeleton import *

print('Maximum likelihood estimation and ODE') 

D = 3 # the dimension of Hawkes processes

# Setting Seqs, which involves setting para and options
para = Parameter()
para.kernel = 'gauss'
para.w = 1.5
para.landmark = np.arange(0, 13, 4) # j:k:n --> np.arange(j, n+1, k), for 0:4:12
L = len(para.landmark)
para.mu = np.matlib.rand(D,1) / D
#para.A = np.zeros((D, D, L))
para.A = np.zeros((L, D, D))

print(para.A)
print(para.A.shape)

for l in range(L):
    # para.A[:,:,l] = (0.5**l) * (0.5 + np.matlib.rand(D, D))
    para.A[l,:,:] = (0.5**l) * (0.5 + np.matlib.rand(D, D))

# para.A = 0.9*para.A./max(abs(eig(sum(para.A,3)))); <--- MATLAB
# para.A = 0.9*para.A / np.max(np.abs(np.linalg.eig(np.sum(para.A, axis=0))))
para.A = 0.9*para.A / np.max(np.abs(np.linalg.eigvals(np.sum(para.A, axis=0))))
para.A = np.reshape(para.A, (D, L, D))

options = Options()
options.N = 200 # the number of sequences
options.Nmax = 100 # the maximum number of events per sequence
options.Tmax = 50 # the maximum size of time window
options.tstep = 0.1
options.dt = 0.1
options.M = 250
options.GenerationNum = 10

Seqs = Simulation_Branch_HP(para, options)

# Setting model
model = Model()
model.M = 1000
model.D = 2
model.dt = 0.02
model.g = np.matlib.rand(model.M, model.D)
model.g = model.g / np.matlib.repmat(np.sum(model.g),[model.M, 1])
model.A = np.matlib.rand(D, model.D, D) / (model.D * D**2)
model.mu = np.matlib.rand(D, 1) / D

# Setting alg
alg = Algorithm()
alg.alpha = 10000
alg.inner = 3
alg.inner_g = 100
alg.outer = 8
alg.thres = 1e-5
alg.Tmax = []

nSeg = 5
nNum = options.N/nSeg
output = Learning_MLE_ODE(Seqs[:i*nNum], model, alg)
print(output)

