# Functions needed for mixed (Basis, ODE) code

import numpy as np

# --------------------------------------------------------------------
# Learning_MLE_Basis.py (now part of Learning_MLE_MIXED.py)
# --------------------------------------------------------------------

def SoftThreshold_LR(A, thres):
	for t in range(size(A,2)):
		tmp = A[:,t,:]
		tmp = np.reshape(tmp, [size(A,1), size(A,3)])
		[Ut, St, Vt] = svd(tmp)
		St = St-thres
		St[St<0] = 0
		obj = Ut * St * np.atleast_2d(Vt).T.conj() # added for Vt' representation
		Z[:,t,:] = obj.reshape(A.shape[0], 1, A.shape[2])
		return Z

def SoftThreshold_S(A, thres):
	tmp = A
	S = np.sign(tmp)
	tmp = (np.abs(tmp)-thres)
	tmp[tmp <= 0] = 0
	Z = [S*tmp]
	return Z

def SoftThreshold_GS(A, thres):
	'''
	Soft-thresholding for group lasso
	'''
	Z = zeros(size(A));
	for u in range(size(A,3)): 
		for v in range(size(A, 1)):
			tmp = 1 - thres/np.linalg.norm(A[v,:,u])
			if tmp > 0:
				Z[v,:,u] = tmp * A[v,:,u]
	return Z

# --------------------------------------------------------------------
# Learning_MLE_ODE.py (now part of Learning_MLE_MIXED.py)
# --------------------------------------------------------------------

def Kernel_Approx(dt, para):
	'''
	Compute the value of kernel function at different time
	dt: array/list (?)
	'''
	g = np.zeros((len(dt.flatten()), para.g.shape[1]))

	print('g:', g)

	M = para.g.shape[0]
	
	print('M:', M)

	Nums = np.ceil(dt/para.dt)

	print('Nums:', Nums)

	#print("Nums------", Nums.shape)
	#print("Nums val", Nums)

	for i in range(len(dt.flatten())):
		#print(Nums[i])
		#if Nums[i] <= M:
		if Nums.flatten()[i] <= M:
			#g[i,:] = para.g[Nums[i].astype(int),:] #g[i,:] is 1x2 initially
			g[i,:] = para.g[int(Nums.flatten()[i]),:]
		else:
			g[i,:] = 0
	return g

def Kernel_Integration_Approx(dt, para):
	#G = np.zeros(len(dt.flatten()), para.g.shape[1])
	G = np.zeros((len(dt.flatten()), para.g.shape[1]))

	#print('G:', G)

	M = para.g.shape[0]

	#print('M:', M)

	Nums = np.ceil(dt/para.dt)

	#print('Nums:', Nums)

	#print(Nums)
	#print("Shape of Nums:", Nums.shape) # added

	for i in range(len(dt.flatten())):
		#if Nums[i] <= M:
		if Nums.flatten()[i] <= M:
			G[i,:] = np.sum(para.g[:int(Nums.flatten()[i]),:]) * para.dt
			print( np.sum(para.g[:int(Nums.flatten()[i]),:]) )
			print("here1")
		else:
			G[i,:] = np.sum(para.g) * para.dt
			print("here2")
	return G

# --------------------------------------------------------------------
# Function tests
# --------------------------------------------------------------------


