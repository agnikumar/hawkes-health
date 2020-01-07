from functions import *
from classes import *

import numpy as np
import numpy.matlib

# MMHP Learning Algorithm
def Learning_MLE_ODE(Seqs, model, alg):
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
			AmatA = AmatA + 2 * alg.alpha * Aest

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

				Nc = len(Time)

				for i in range(Nc):
					ui = Event[i]

					AmatA[ui,:,:] += (Aest[ui,:,:] > 0) * np.tile(GK[i,:], [1,1,D]) # check if np.matlib.repmat could be used
					
					ti = Time[i]

					if n == alg.inner:
						Nums = min([np.ceil(dT[i]/model.dt), CM.shape[0]])
						CM[:Nums,:] += np.matlib.repmat(sum(Aest[ui,:,:], 3), Nums, 1) 

					lambdai = muest[ui]
					pii = muest[ui]
					pij = []

					if i > 1:
						tj = Time[:i-1]
						uj = Event[:i-1]

						dt = ti - tj
						gij = Kernel_Approx(dt, model) 
						auiuj = Aest[uj,:,ui]
						pij = auiuj * gij
						lambdai = lambdai + np.sum(pij.flatten()) 

					NLL = NLL - np.log(lambdai)
					pii = pii/lambdai 

					if i > 1:
						pij = pij/lambdai
						if (pij.size != 0) and (np.sum(pij.flatten()) > 0): 
							for j in range(len(uj)):
								uuj = uj[j]
								BmatA[uuj,:,ui] += Aest[uuj,:,ui] * pij[j,:]

								if n == alg.inner:
									Nums = min([np.ceil(dt[j]/model.dt), DM.shape[0]])
									DM[Nums,:] += pij[j,:]

					Bmu[ui] += pii

				NLL += (Tstop - Tstart) * np.sum(muest)
				NLL += np.sum(np.sum(GK * np.sum(Aest[Event,:,:], 3)))

			# M-step: update parameters
			mu = Bmu/Amu
			A = np.abs(np.sqrt(BmatA/AmatA))    
			A[isnan(A)] = 0
			A[isinf(A)] = 0

			Err = np.sum(np.sum(np.sum(np.abs(A-Aest))))/np.sum(np.abs(Aest.flatten()))
			Aest = A
			muest = mu
			model.A = Aest
			model.mu = muest

			print('Outer={}, Inner={}, Objective={}, RelErr={}'.format(o, n, NLL, Err)) # edit 

		DM = DM/model.dt
		CM = CM/model.g
		gest = model.g
		for n in range(alg.inner_g):
			for m in range(model.g.shape[0]):
				if m == 1:
					a = 2 * alg.alpha + CM[m,:] * (model.dt**2)
					b = -2 * alg.alpha * model.g[m+1,:]
					c = -DM[m,:] * (model.dt**2)
				elif m == model.g.shape[0]: 
					a = 4 * alg.alpha + CM[m,:] * (model.dt**2)
					b = -2 * alg.alpha * model.g[m-1,:]
					c = -DM[m,:] * (model.dt**2)
				else:
					a = 4 * alg.alpha + CM[m,:] * (model.dt**2)
					b = -2 * alg.alpha * (model.g[m-1,:] + model.g[m+1,:])
					c = -DM[m,:] * (model.dt**2)

			gest[m,:] = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

		model.g = np.abs(gest)
		model.g = model.g/np.matlib.repmat(np.sum(model.g), model.M, 1) 

	#return model
	return (model.A, model.mu, model.g)

def Loglike_HP_ODE(Seqs, model, alg):
	'''
	Contained in Analysis folder
	'''
	Aest = model.A
	muest = model.mu

	for c in range(len(Seqs)):
		Time = Seqs[c].Time 
		Event = Seqs[c].Mark 
		Tstart = Seqs[c].Start

		if alg.Tmax.size == 0: 
			Tstop = Seqs[c].Stop
		else:
			Tstop = alg.Tmax
			indt = Time < alg.Tmax
			Time = Time[indt] 
			Event = Event[indt] 

		Amu = Amu + Tstop - Tstart 

		dT = Tstop - Time
		GK = Kernel_Integration_Approx(dT, model)

		Nc = len(Time)

		for i in range(Nc):
			ui = Event[i]
			ti = Time[i]
			lambdai = muest[ui]

			if i > 1:
				tj = Time[:i-1]
				uj = Event[:i-1]

				dt = ti - tj
				gij = Kernel_Approx(dt, model) 
				auiuj = Aest[uj,:,ui]
				pij = auiuj * gij
				lambdai = lambdai + np.sum(pij.flatten()) 

			Loglike = Loglike - np.log(lambdai) # check this (first mention of Loglike)

		Loglike = Loglike + (Tstop - Tstart) * np.sum(muest);
		Loglike = Loglike + np.sum(np.sum(GK * np.sum(Aest[Event,:,:], 3)))

	Loglike = -Loglike

	return Loglike



