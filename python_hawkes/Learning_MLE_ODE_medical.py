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
	
	########

	for o in range(alg.outer):
		DM = np.zeros(model.g.shape)
		CM = DM # CM is 1000x41

		#########

		for n in range(alg.inner): ##### Aest breaks down to zeros?, log print happens here
			#print("Aest here:", Aest) # ADDED

			NLL = 0 # negative log-likelihood

			Amu = np.zeros((D, 1))
			Bmu = Amu

			BmatA = np.zeros(Aest.shape)
			AmatA = BmatA
			#AmatA = AmatA + 2 * alg.alpha * Aest #3x2x3
			AmatA = np.add(AmatA, 2 * alg.alpha * Aest) # EDITED

			#print("BmatA initial:", BmatA) # TRIAL
			#print("AmatA initial:", AmatA) # TRIAL

			#print("Aest:", Aest)
			#print("-------------")
			#print("AmatA here:", AmatA)

			# E-step: evaluate the responsibility using the current parameters    
			for c in range(len(Seqs)):
				Time = Seqs[c].Time 

				#print("Time:", Time) # ADDED
				#print(Time.shape)

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
				#print("dT shapeeeeee", dT.shape) #e.g., 1x29
				#print("dt element:", dT.flatten()[1])
				#dT_original = Tstop - Time
				#dT = np.reshape(Tstop - Time, (1, dT_original.shape[0]))

				#print("dT:", dT)
				#print(dT.shape)

				GK = Kernel_Integration_Approx(dT, model) 

				#print("GK:", GK)
				#print(GK.shape)

				#Nc = len(Time)
				Nc = Time.shape[1]

				for i in range(Nc):
					#print("i=", i)
					#ui = Event[i]
					ui = int(Event.flatten()[i])  # EDITED -1

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
					

					#print("AmatA shape:", AmatA.shape)
					#print("ui:", ui)
					#print("AmatA[ui,:,:].shape before", AmatA[ui,:,:].shape)


					#AmatA[ui,:,:] += (Aest[ui,:,:] > 0) * np.dstack([GK[i,:]]*D) #* np.tile(GK_temp[:,:,np.newaxis], (1,1,D))
					#temp_AmatA_slice = np.reshape(AmatA[ui,:,:], (1, shape_r, shape_c)) \
					#			+ (np.reshape((Aest[ui,:,:] > 0), (1, shape_r, shape_c))) * np.dstack([GK[i,:]]*D)
					#AmatA[ui,:,:] = np.reshape(temp_AmatA_slice, (shape_r, shape_c))

					
					temp_AmatA_slice = np.reshape(AmatA[ui,:,:], (1, shape_r, shape_c)) \
								+ (np.reshape((Aest.copy()[ui,:,:] > 0), (1, shape_r, shape_c))) * np.dstack([GK[i,:]]*D)
					AmatA[ui,:,:] = np.reshape(temp_AmatA_slice, (1, shape_r, shape_c))
					#print("AmatA after update:", AmatA, AmatA.shape) # TRIAL

					#print("AmatA[ui,:,:].shape afterwards", AmatA[ui,:,:].shape)
					#AmatA[ui,:,:] += (Aest[ui,:,:] > 0) * np.dstack([GK[i,:]]*D)

					#ti = Time[i]
					ti = Time.flatten()[i]
					#print("ti:", ti)

					if n == alg.inner - 1: # EDITED -1
						#Nums = min([np.ceil(dT[i]/model.dt), CM.shape[0]])
						Nums = min([np.ceil(dT.flatten()[i]/model.dt), CM.shape[0]])
						#print("Nums:", Nums)
						#CM[:Nums,:] += np.matlib.repmat(sum(Aest[ui,:,:], 3), Nums, 1) 
						CM[:int(Nums),:] += np.tile(np.sum(Aest[ui,:,:], 0), (int(Nums), 1)) #should axis be 0 or 1?
						#print("CM[:int(Nums),:].shape", CM[:int(Nums),:].shape)

					lambdai = muest[ui]
					pii = muest[ui]
					pij = np.empty((0,0))


					if i > 0: #if i > 1:
						tj = Time.flatten()[:i-1]
						uj = Event.flatten()[:i-1]

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
						#print()

						#print("dt_shape", dt.shape) # ADDED
						#print(uj) # added
						auiuj = Aest[uj.astype(int), :, ui]
						pij = auiuj * gij
						#print("pij_shape", pij.shape) # ADDED
						lambdai = lambdai + np.sum(pij.flatten()) 

					NLL = NLL - np.log(lambdai)
					pii = pii/lambdai 

					if i > 0: #if i > 1:
						pij = pij/lambdai
						#print("pij", pij)
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
								
								#temp_BmatA_slice = np.reshape(BmatA.copy()[uuj,:,ui], (1,2)) + np.multiply(np.reshape(Aest[uuj,:,ui], (1,2)), pij[j,:])
								#BmatA[uuj,:,ui] = np.reshape(temp_BmatA_slice, (1,2))
								shape_reshape = Aest[uuj,:,ui].shape[0] # number of units
								temp_BmatA_slice = np.reshape(BmatA.copy()[uuj,:,ui], (1, shape_reshape)) + np.multiply(np.reshape(Aest[uuj,:,ui], (1, shape_reshape)), pij[j,:])
								BmatA[uuj,:,ui] = np.reshape(temp_BmatA_slice, (1, shape_reshape))
								#print("BmatA after update:", BmatA, BmatA.shape) # TRIAL

								if n == alg.inner:
									Nums = min([np.ceil(dt[j]/model.dt), DM.shape[0]])
									DM[Nums,:] += pij[j,:]

					#print(Bmu.shape) # added
					#print(Bmu[ui].shape) # added
					#Bmu[ui] += pii
					#Bmu[ui] += np.reshape(pii, (1,)) # last occurrence of pii
					Bmu[ui] = np.reshape(Bmu[ui], (1,1)) + pii #range(Nc) loop stops

				NLL += (Tstop - Tstart) * np.sum(muest)
				#print(Event.shape)
				NLL += np.sum(np.sum(GK * np.sum(Aest[Event,:,:], 2))) # should this be a 3?

				#print("BmatA:", BmatA) # ADDED
				#print("AmatA:", AmatA) # ADDED

			# M-step: update parameters
			mu = Bmu/Amu

			#print("BmatA:", BmatA) # ADDED
			#print("AmatA:", AmatA) # ADDED
			#print("BmatA later on:", BmatA) # TRIAL
			#print("AmatA later on:", AmatA) # TRIAL

			A = np.abs(np.sqrt(BmatA/AmatA))  
			#print("A:", A) # ADDED  
			A[np.isnan(A)] = 0
			A[np.isinf(A)] = 0

			#print("A:", A) # ADDED

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

			#print()
			#stuff = (-b + np.sqrt(np.square(b) - 4*a*c))/(2*a) # ADDED
			#print("lots of stuff shape:", stuff.shape) # ADDED

			#gest[m,:] = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
			gest[m,:] = (-b + np.sqrt(np.square(b) - 4*np.multiply(a,c)))/(2*a)

		model.g = np.abs(gest)
		model.g = model.g/np.matlib.repmat(np.sum(model.g), model.M, 1) 

	#return model
	return (model.A, model.mu, model.g, NLL[0])
