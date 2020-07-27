from functions import *
from classes import *
from classes_skeleton import *

import numpy as np
import numpy.matlib

# MMHP Learning Algorithm
def Learning_MLE_ODE_medical(Seqs, model, alg):
	'''
	Learning Hawkes processes via maximum likelihood estimation 
	with the help of ordinary differential equations (ODE)
	Seqs: array/list of instances of the Sequence class
	'''
	Aest = model.A
	muest = model.mu
	D = Aest.shape[0]

	for o in range(alg.outer):
		DM = np.zeros(model.g.shape)
		CM = DM 

		for n in range(alg.inner): 

			NLL = 0 # negative log-likelihood

			Amu = np.zeros((D, 1))
			Bmu = Amu

			BmatA = np.zeros(Aest.shape)
			AmatA = BmatA
			AmatA = np.add(AmatA, 2 * alg.alpha * Aest) 

			# E-step: evaluate the responsibility using the current parameters    
			for c in range(len(Seqs)):
				Time = Seqs[c].Time 
				Event = Seqs[c].Mark 
				Tstart = Seqs[c].Start 

				if alg.Tmax.size == 0: # check if array is empty
					Tstop = Seqs[c].Stop
				else:
					Tstop = alg.Tmax
					indt = Time < alg.Tmax
					Time = Time[indt] 
					Event = Event[indt] 

				Amu = Amu + Tstop - Tstart
				dT = Tstop - Time
				GK = Kernel_Integration_Approx(dT, model) 
				Nc = Time.shape[1]

				for i in range(Nc):
					ui = int(Event.flatten()[i]) 
					
					shape_r = AmatA[ui,:,:].shape[0]
					shape_c = AmatA[ui,:,:].shape[1]
					temp_AmatA_slice = np.reshape(AmatA[ui,:,:], (1, shape_r, shape_c)) \
								+ (np.reshape((Aest.copy()[ui,:,:] > 0), (1, shape_r, shape_c))) * np.dstack([GK[i,:]]*D)
					AmatA[ui,:,:] = np.reshape(temp_AmatA_slice, (1, shape_r, shape_c))

					ti = Time.flatten()[i]

					if n == alg.inner - 1: 
						Nums = min([np.ceil(dT.flatten()[i]/model.dt), CM.shape[0]])
						CM[:int(Nums),:] += np.tile(np.sum(Aest[ui,:,:], 0), (int(Nums), 1)) # check axis

					lambdai = muest[ui]
					pii = muest[ui]
					pij = np.empty((0,0))

					if i > 0: 
						tj = Time.flatten()[:i-1]
						uj = Event.flatten()[:i-1]

						dt_original = ti - tj
						dt = np.reshape(ti - tj, (1, dt_original.shape[0])) 

						gij = Kernel_Approx(dt, model) 

						auiuj = Aest[uj.astype(int), :, ui]
						pij = auiuj * gij

						lambdai = lambdai + np.sum(pij.flatten()) 

					NLL = NLL - np.log(lambdai)
					pii = pii/lambdai 

					if i > 0: 
						pij = pij/lambdai
						if (pij.size != 0) and (np.sum(pij.flatten()) > 0): 
							for j in range(len(uj)):
								uuj = int(uj[j])

								shape_reshape = Aest[uuj,:,ui].shape[0] # number of units
								temp_BmatA_slice = np.reshape(BmatA.copy()[uuj,:,ui], (1, shape_reshape)) + np.multiply(np.reshape(Aest[uuj,:,ui], (1, shape_reshape)), pij[j,:])
								BmatA[uuj,:,ui] = np.reshape(temp_BmatA_slice, (1, shape_reshape))
				
								if n == alg.inner:
									Nums = min([np.ceil(dt[j]/model.dt), DM.shape[0]])
									DM[Nums,:] += pij[j,:]

					Bmu[ui] = np.reshape(Bmu[ui], (1,1)) + pii 

				NLL += (Tstop - Tstart) * np.sum(muest)
				NLL += np.sum(np.sum(GK * np.sum(Aest[Event,:,:], 2))) # check axis
			# M-step: update parameters
			mu = Bmu/Amu

			A = np.abs(np.sqrt(BmatA/AmatA))  
			A[np.isnan(A)] = 0
			A[np.isinf(A)] = 0

			Err = np.sum(np.sum(np.sum(np.abs(A-Aest))))/np.sum(np.abs(Aest.flatten()))
			Aest = A
			muest = mu
			model.A = Aest
			model.mu = muest

			print('Outer={}, Inner={}, Objective={}, RelErr={}'.format(o, n, NLL, Err)) 

		DM = DM/model.dt
		CM = CM/model.g
		gest = model.g
		for n in range(alg.inner_g):
			for m in range(model.g.shape[0]):
				if m == 0:
					a = 2 * alg.alpha + CM[m,:] * (model.dt**2)
					b = -2 * alg.alpha * model.g[m+1,:]
					c = -DM[m,:] * (model.dt**2)
				elif m == model.g.shape[0] - 1: 
					a = 4 * alg.alpha + CM[m,:] * (model.dt**2)
					b = -2 * alg.alpha * model.g[m-1,:]
					c = -DM[m,:] * (model.dt**2)
				else:
					a = 4 * alg.alpha + CM[m,:] * (model.dt**2)
					b = -2 * alg.alpha * (model.g[m-1,:] + model.g[m+1,:])
					c = -DM[m,:] * (model.dt**2)

			gest[m,:] = (-b + np.sqrt(np.square(b) - 4*np.multiply(a,c)))/(2*a)

		model.g = np.abs(gest)
		model.g = model.g/np.matlib.repmat(np.sum(model.g), model.M, 1) 

	#return model
	return (model.A, model.mu, model.g, NLL[0])
