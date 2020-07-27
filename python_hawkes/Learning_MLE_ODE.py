from functions import *
from classes import *
from classes_skeleton import *

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
			AmatA = AmatA + 2 * alg.alpha * Aest #3x2x3

			# E-step: evaluate the responsibility using the current parameters    
			for c in range(len(Seqs)):
				Time = Seqs[c].Time 

				print("Time:", Time) # ADDED
				print(Time.shape)

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

				dT_original = Tstop - Time
				dT = np.reshape(Tstop - Time, (1, dT_original.shape[0]))

				#print("dT:", dT)
				#print(dT.shape)

				GK = Kernel_Integration_Approx(dT, model) 

				#print("GK:", GK)
				#print(GK.shape)

				Nc = len(Time)

				for i in range(Nc):
					#ui = Event[i]
					ui = int(Event[i])

					#AmatA(ui,:,:) = AmatA(ui,:,:)+...
                    #double(Aest(ui,:,:)>0).*repmat( GK(i,:), [1,1,D] ); <--- MATLAB

					#AmatA[ui,:,:] += (Aest[ui,:,:] > 0) * np.tile(GK[i,:], [1,1,D]) # check if np.matlib.repmat could be used
					#print("GK", GK)
					#print(GK.shape)
					#GK_adjusted = M[:,:,np.newaxis] # added, remove later
					#GK_temp = GK[i,:] #1x2
					#obj = Aest #np.dstack([GK_temp]*D) # ADDED
					#print("----", obj.shape)

					# AmatA[ui,:,:] shape
					shape_r = AmatA[ui,:,:].shape[0]
					shape_c = AmatA[ui,:,:].shape[1]

					#AmatA[ui,:,:] += (Aest[ui,:,:] > 0) * np.dstack([GK_temp]*D) #* np.tile(GK_temp[:,:,np.newaxis], (1,1,D))
					temp_AmatA_slice = np.reshape(AmatA[ui,:,:], (1, shape_r, shape_c)) \
								+ (np.reshape((Aest[ui,:,:] > 0), (1, shape_r, shape_c))) * np.dstack([GK[i,:]]*D)
					AmatA[ui,:,:] = np.reshape(temp_AmatA_slice, (shape_r, shape_c))

					ti = Time[i]

					if n == alg.inner:
						Nums = min([np.ceil(dT[i]/model.dt), CM.shape[0]])
						CM[:Nums,:] += np.matlib.repmat(sum(Aest[ui,:,:], 3), Nums, 1) 

					lambdai = muest[ui]
					pii = muest[ui]
					pij = np.empty((0,0))


					if i > 1:
						tj = Time[:i-1]
						uj = Event[:i-1]

						#print("i:", i)
						#print("ti", ti)
						#print("tj", tj)
						#print("--------------")

						#dt = ti - tj
						dt_original = ti - tj
						#print(dt.shape) # ADDED
						dt = np.reshape(ti - tj, (1, dt_original.shape[0])) # ADDED
						#print("dt shape:", dt.shape) # ADDED
						#print("dt val:", dt)

						gij = Kernel_Approx(dt, model) 

						#print("dt_shape", dt.shape) # ADDED
						#print(uj) # added
						auiuj = Aest[uj.astype(int), :, ui]
						pij = auiuj * gij
						#print("pij_shape", pij.shape) # ADDED
						lambdai = lambdai + np.sum(pij.flatten()) 

					NLL = NLL - np.log(lambdai)
					pii = pii/lambdai 

					if i > 1:
						pij = pij/lambdai
						if (pij.size != 0) and (np.sum(pij.flatten()) > 0): 
							for j in range(len(uj)):
								uuj = int(uj[j])

								#print("pij_shape", pij.shape) # ADDED
								#print(np.reshape(Aest[uuj,:,ui], (1,2)))

								#BmatA[uuj,:,ui] += Aest[uuj,:,ui] * pij[j,:] # Aest[uuj,:,ui] is 1x2
								#BmatA[uuj,:,ui] += np.reshape(Aest[uuj,:,ui], (1,2)) * pij[j,:]
								#print(BmatA.shape)
								#print(BmatA[uuj,:,ui].shape)
								#print(BmatA[uuj,:,ui])
								#BmatA[uuj,:,ui] += np.multiply(np.reshape(Aest[uuj,:,ui], (1,2)), pij[j,:])
								temp_BmatA_slice = np.reshape(BmatA[uuj,:,ui], (1,2)) + np.multiply(np.reshape(Aest[uuj,:,ui], (1,2)), pij[j,:])
								BmatA[uuj,:,ui] = np.reshape(temp_BmatA_slice, (1,2))

								if n == alg.inner:
									Nums = min([np.ceil(dt[j]/model.dt), DM.shape[0]])
									DM[Nums,:] += pij[j,:]

					#print(Bmu.shape) # added
					#print(Bmu[ui].shape) # added
					#Bmu[ui] += pii
					#Bmu[ui] += np.reshape(pii, (1,)) # last occurrence of pii
					Bmu[ui] = np.reshape(Bmu[ui], (1,1)) + pii 

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



