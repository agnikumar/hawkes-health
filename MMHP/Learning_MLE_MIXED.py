# Mixed (Basis, ODE) code

from functions_MIXED import *
from classes_MIXED import *

import numpy as np

def Learning_MLE_MIXED(Seqs, model, alg):

	# initial
	Aest = model.A
	muest = model.mu

	if alg.LowRank:
		UL = np.zeros(Aest.shape)
		ZL = Aest

	if alg.Sparse:
		US = np.zeros(Aest.shape)
		ZS = Aest

	if alg.GroupSparse:
		UG = np.zeros(Aest.shape)
		ZG = Aest

	D = Aest.shape[0]

	if alg.storeLL:
		model.LL = np.zeros((alg.outer, 1))

	if alg.storeErr:
		model.err = np.zeros((alg.outer, 3))

	for o in range(alg.outer):
		DM = np.zeros(model.g.shape)
		CM = DM
		rho = alg.rho * (1.1**o)

		for n in range(alg.inner): 
			NLL = 0 # negative log-likelihood
			Amu = np.zeros((D, 1))
			Bmu = Amu
			
			CmatA = np.zeros(Aest.shape)
			AmatA = CmatA
			BmatA = CmatA
			if alg.LowRank:
				BmatA = BmatA + rho*(UL-ZL)
				AmatA = AmatA + rho

			if alg.Sparse:
				BmatA = BmatA + rho*(US-ZS)
				AmatA = AmatA + rho

			if alg.GroupSparse:
				BmatA = BmatA + rho*(UG-ZG)
				AmatA = AmatA + rho

			# E-step: evaluate the responsibility using the current parameters    
			for c in range(len(Seqs)):
				if Seqs[c].Time.size != 0: # ~isempty
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
					print(GK)

					Nc = len(Time)
					
					for i in range(Nc):
						ui = Event[i]
						#print(ui)

						#BmatA[ui,:,:] = BmatA[ui,:,:] + float(Aest[ui,:,:]>0) * np.matlib.repmat(GK[i,:], [1,1,D])
						#STUFF = Aest[ui,:,:]>0
						#print(STUFF)
						#print(STUFF.shape)
						BmatA[ui,:,:] += (Aest[ui,:,:]>0) * np.dstack(GK[i,:]*D)
						# shape_r = BmatA[ui,:,:].shape[0]
						# shape_c = BmatA[ui,:,:].shape[1]
						# temp_BmatA_slice = np.reshape(BmatA[ui,:,:], (1, shape_r, shape_c)) \
						# 			+ (np.reshape((Aest.copy()[ui,:,:] > 0), (1, shape_r, shape_c))) * np.dstack([GK[i,:]]*D)
						# BmatA[ui,:,:] = np.reshape(temp_BmatA_slice, (1, shape_r, shape_c))

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
									CmatA[uuj,:,ui] = CmatA[uuj,:,ui] - pij[j,:]

									if n == alg.inner:
										Nums = min([np.ceil(dt[j]/model.dt), DM.shape[0]])
										DM[Nums,:] += pij[j,:]

						Bmu[ui] = Bmu[ui] + pii

					NLL = NLL + (Tstop - Tstart) * np.sum(muest)
					NLL = NLL + np.sum(np.sum(GK * np.sum(Aest[Event,:,:], 3)))

				else:
					print('Sequence {} is empty!'.format(c))

			# M-step: update parameters
			mu = Bmu / Amu       
			if alg.Sparse==0 and alg.GroupSparse==0 and alg.LowRank==0:
				A = -CmatA / BmatA #(-BA + np.sqrt(BA**2 - 4*AA*CA)) / (2*AA)
				A[np.isnan(A)] = 0
				A[np.isinf(A)] = 0
			else:            
				A = (-BmatA + np.sqrt(BmatA**2 - 4*AmatA*CmatA))/(2*AmatA)
				A[np.isnan(A)] = 0
				A[np.isinf(A)] = 0

			# check convergence
			Err = np.sum(np.sum(np.sum(np.abs(A-Aest))))/np.sum(np.abs(Aest.flatten()))
			Aest = A
			muest = mu
			model.A = Aest
			model.mu = muest
			print('Outer={}, Inner={}, Objective={}, RelErr={}'.format(o, n, NLL, Err)) # missing toc
				
			if Err < alg.thres or (o==alg.outer and n==alg.inner):
				break

			# store loglikelihood
			# if alg.storeLL:
			#     Loglike = Loglike_Basis(Seqs, model, alg) 
			#     model.LL[o] = Loglike

		# calculate error:
		# if alg.storeErr:
		# 	Err = np.zeros(1,3)
		# 	Err[1] = np.linalg.norm(model.mu.flatten() - alg.truth.mu.flatten())/np.linalg.norm(alg.truth.mu.flatten())
		# 	Err[2] = np.linalg.norm(model.A.flatten() - alg.truth.A.flatten())/np.linalg.norm(alg.truth.A.flatten())
		# 	Err[3] = np.linalg.norm([model.mu.flatten(), model.A.flatten()]-[alg.truth.mu.flatten(), alg.truth.A.flatten()]) \
		# 		/np.linalg.norm([alg.truth.mu.flatten(), alg.truth.A.flatten()])
		# 	model.err[o-1,:].copy() = Err
		
		if alg.LowRank:
			threshold = alg.alphaLR/rho
			ZL = SoftThreshold_LR(Aest+UL, threshold) 
			UL = UL + (Aest-ZL)
		
		if alg.Sparse:
			threshold = alg.alphaS/rho
			ZS = SoftThreshold_S(Aest+US, threshold) 
			US = US + (Aest-ZS)

		if alg.GroupSparse:
			threshold = alg.alphaGS/rho
			ZG = SoftThreshold_GS(Aest+UG, threshold) 
			UG = UG + (Aest-ZG)

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

