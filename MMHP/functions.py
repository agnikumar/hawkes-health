import numpy as np

# --------------------------------------------------------------------
# Learning_MLE_Basis.py 
# --------------------------------------------------------------------

def Loglike_Basis(Seqs, model, alg):
    # initial 
    Aest = model.A      
    muest = model.mu

    # D = Aest.shape[0]
 
    Loglike = 0 # negative log-likelihood

    # E-step: evaluate the responsibility using the current parameters    
    for c in range(len(Seqs)):
        Time = Seqs[c].Time
        Event = Seqs[c].Mark
        Tstart = Seqs[c].Start

        if alg.Tmax == 0:
            Tstop = Seqs[c].Stop
        else:
            Tstop = alg.Tmax
            indt = Time < alg.Tmax
            Time = Time[indt]
            Event = Event[indt]

        # Amu = Amu + Tstop - Tstart

        dT = Tstop - Time
        GK = Kernel_Integration(dT, model) # TODO: code Kernel_Integration

        Nc = len(Time)

        for i in range(Nc):
            ui = Event[i]
            ti = Time[i]           

            lambdai = muest[ui]
            #pii = muest[ui]
            #pij = []

            if i > 1:

                tj = Time[:i-1]
                uj = Event[:i-1]

                dt = ti - tj
                gij = Kernel(dt, model)
                auiuj = Aest[uj, :, ui]
                pij = auiuj * gij
                lambdai = lambdai + np.sum(pij.flatten())

            Loglike = Loglike - np.log(lambdai)

        Loglike = Loglike + (Tstop-Tstart) * sum(muest)
        Loglike = Loglike + np.sum(np.sum(GK * np.sum(Aest[Event,:,:],3)))

    Loglike = -Loglike
    return Loglike

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
# Learning_MLE_ODE.py 
# --------------------------------------------------------------------

def Kernel_Approx(dt, para):
    '''
    Compute the value of kernel function at different time
    dt: array/list (?)
    '''
    g = np.zeros((len(dt.flatten()), para.g.shape[1]))

    M = para.g.shape[0]
    Nums = np.ceil(dt/para.dt)

    #print("Nums------", Nums.shape)
    #print("Nums val", Nums)

    for i in range(len(dt.flatten())):
        #print(Nums[i])
        #if Nums[i] <= M:
        if Nums.flatten()[i] < M:
            #g[i,:] = para.g[Nums[i].astype(int),:] #g[i,:] is 1x2 initially
            g[i,:] = para.g[int(Nums.flatten()[i]),:]
        else:
            g[i,:] = 0
    return g

def Kernel_Integration_Approx(dt, para):
    #G = np.zeros(len(dt.flatten()), para.g.shape[1])
    G = np.zeros((len(dt.flatten()), para.g.shape[1]))

    M = para.g.shape[0]
    Nums = np.ceil(dt/para.dt)

    #print(Nums)
    #print("Shape of Nums:", Nums.shape) # added

    for i in range(len(dt.flatten())):
        #if Nums[i] <= M:
        if Nums.flatten()[i] < M:
            G[i,:] = np.sum(para.g[:int(Nums.flatten()[i]),:]) * para.dt
        else:
            G[i,:] = np.sum(para.g) * para.dt
    return G

# --------------------------------------------------------------------
# MMHP_testing.py / Simulation_Branch_HP.py
# --------------------------------------------------------------------
def Simulation_Thinning_Poisson(mu, t_start, t_end):
    '''
    Thinning method to simulate homogeneous Poisson processes
    '''

    #print("mu:", mu)
    #print(mu.shape)
    #print("t_start:", t_start)
    #print("t_end:", t_end)

    t = t_start
    #History = []
    History = np.empty((2,0))
    #print(History.shape)

    mt = np.sum(mu)

    while t < t_end:
        s = np.random.exponential(mt) # check correctness 
        t = t + s

        u = np.random.uniform(0, 1) * mt 
        sumIs = 0
        for d in range(len(mu)): 
            sumIs = sumIs + mu[d] 
            if sumIs >= u:
                break

        index = d
        # History=[History,[t;index(1)]]; <--- MATLAB
        #History = np.concatenate((History, np.concatenate(t, index[0])), 1) # check correctness
        #History = np.concatenate((History, np.concatenate((t, index))), 1) # check correctness
        t_index = np.vstack((t, index))

        #print(History.shape)
        #print("History:", History)
        #print("t_index:", t_index)
        #print(t_index.shape)

        History = np.hstack((History, t_index))

    index = np.nonzero(History[0,:].copy() < t_end) 
    #print(index)
    #print(index[0].shape)
    #History = History[:, index] 
    History = History[:, index[0]] 
    #print("History shape:", History.shape)
    return History

def Kernel(dt, para): 
    '''
    Compute the value of kernel function at different time
    Used in ImpactFunction
    '''
    #distance = np.matlib.repmat(dt.flatten(), 1, len(para.landmark.flatten())) - \
    #        np.matlib.repmat(np.atleast_2d(para.landmark.flatten()).T, len(dt), 1)
    #distance = np.matlib.repmat(dt, 1, len(para.landmark.flatten())) - \
    #        np.matlib.repmat(np.atleast_2d(para.landmark.flatten()).T, 1, 1) # should be 1x4, but is 4x4
    distance = np.tile(dt, (len(para.landmark.flatten()), 1)) - \
            np.tile(np.atleast_2d(para.landmark.flatten()).T, (1, 1))

    #print("distance", distance.shape) # added

    if para.kernel == 'exp':
        g = para.w * np.exp(-para.w * distance)
        g[g > 1] = 0 
    elif para.kernel == 'gauss':
        #g = np.exp(-(distance**2)/(2*para.w**2))/(np.sqrt(2*np.pi)*para.w) 
        g_temp = np.exp(-(distance**2)/(2*para.w**2))/(np.sqrt(2*np.pi)*para.w)
        g = g_temp.reshape((1,4))
        #print("g", g) # added
        #print(g.shape) # added
    else:
        print('Error: please assign a kernel function!')
        g = None # added 
    return g

def ImpactFunction(u, dt, para):
    #print(u)
    #print(u.shape)
    #A = reshape(para.A(u,:,:), [size(para.A, 2), size(para.A, 3)]); <--- MATLAB
    #A = np.reshape(para.A[u,:,:], size(para.A, 2), size(para.A, 3))
    #print(para.A.shape) # should be 3x4x3

    #print(para.A[int(u),:,:])
    #print(para.A[int(u),:,:].shape)

    A = np.reshape(para.A[int(u),:,:], (para.A.shape[1], para.A.shape[2]), order='F').copy() # should be 4x3

    #print(A.shape) # should be 4x3

    basis = Kernel(dt, para)

    #print("Basis", basis.shape) # should be 1x4

    #phi = np.atleast_2d(A).T * basis.flatten()
    #print(A.conj().T.shape)
    #print("basis flatten", basis.flatten().shape)
    #print(basis.flatten())

    phi = A.conj().T @ basis.flatten().reshape((4,1))
    #print("phi shape", phi.shape) # is 3x4
    return phi



