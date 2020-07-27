# Learning Hawkes processes via maximum likelihood estimation
# Different regularizers (low-rank, sparse, group sparse) of parameters and
# their combinations are considered, which are solved via ADMM.

from functions import *
from classes import *
import numpy as np

def Learning_MLE_Basis(Seqs, model, alg):
    '''
    # Learning Hawkes processes via maximum likelihood estimation
    # Different regularizers (low-rank, sparse, group sparse) of parameters and
    # their combinations are considered, which are solved via ADMM.
    '''

    # initial 
    Aest = model.A      
    muest = model.mu

    #GK = struct('intG', [])

    if alg.LowRank:
        UL = np.zeros(shape(Aest))
        ZL = Aest

    if alg.Sparse
        US = np.zeros(shape(Aest))
        ZS = Aest

    #------------------------------------- 
    if alg.GroupSparse 
        UG = np.zeros(shape(Aest))
        ZG = Aest
    #-------------------------------------

    D = Aest.shape[0]

    if alg.storeLL:
        model.LL = np.zeros(alg.outer, 1)

    if alg.storeErr:
        model.err = np.zeros(alg.outer, 3)

    #tic
    for o in range(alg.outer):
        
        rho = alg.rho * (1.1**o)
        
        for n in range(alg.inner):   
            
            NLL = 0 # negative log-likelihood
            
            Amu = np.zeros(D, 1)
            Bmu = Amu
            
            
            CmatA = np.zeros(shape(Aest))
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
            for c in range(len(Seqs))
                if Seqs[c].Time.size == 0: # ~isempty
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
                    GK = Kernel_Integration(dT, model)
    #                 if o==1:
    #                     GK(c).intG = Kernel_Integration(dT, model)

                    Nc = len(Time)
                    
                    for i in range(Nc):

                        ui = Event[i]

                        BmatA[ui,:,:] = BmatA[ui,:,:] + float(Aest[ui,:,:]>0) * np.matlib.repmat(GK[i,:], [1,1,D])

                        ti = Time[i]           

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
                            if pij.size != 0 and np.sum(pij.flatten()) > 0
                                for j in range(len(uj)):
                                    uuj = uj[j]
                                    CmatA[uuj,:,ui] = CmatA[uuj,:,ui] - pij[j,:]

                        Bmu[ui] = Bmu[ui] + pii

                    NLL = NLL + (Tstop - Tstart) * np.sum(muest)
                    NLL = NLL + np.sum(np.sum(GK * np.sum(Aest[Event,:,:], 3)))
                    #NLL = NLL + np.sum(np.sum(GK[c].intG * np.sum(Aest[Event,:,:],3)))
                
                else:
                    print('Sequence {} is empty!'.format(c))

                    
            # M-step: update parameters
            mu = Bmu / Amu       
            if alg.Sparse==0 and alg.GroupSparse==0 and alg.LowRank==0:
                A = -CmatA / BmatA #(-BA + np.sqrt(BA**2 - 4*AA*CA)) / (2*AA)
                A[isnan(A)] = 0
                A[isinf(A)] = 0
            else:            
                A = (-BmatA + np.sqrt(BmatA**2 - 4*AmatA*CmatA))/(2*AmatA)
                A[isnan(A)] = 0
                A[isinf(A)] = 0
             
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
        if alg.storeLL:
            Loglike = Loglike_Basis(Seqs, model, alg) 
            model.LL[o] = Loglike

        # calculate error:
        if alg.storeErr:
            Err = np.zeros(1,3)
            Err[1] = np.linalg.norm(model.mu.flatten() - alg.truth.mu.flatten())/np.linalg.norm(alg.truth.mu.flatten())
            Err[2] = np.linalg.norm(model.A.flatten() - alg.truth.A.flatten())/np.linalg.norm(alg.truth.A.flatten())
            Err[3] = np.linalg.norm([model.mu.flatten(), model.A.flatten()]-[alg.truth.mu.flatten(), alg.truth.A.flatten()]) \
                /np.linalg.norm([alg.truth.mu.flatten(), alg.truth.A.flatten()])
            model.err[o-1,:].copy() = Err
        
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

    return model
