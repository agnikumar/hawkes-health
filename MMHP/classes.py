# Characterizing model
class Model:
	def __init__(self, A=None, mu=None, g=None, dt=None, M=None, LL=None, err=None):
		
		# both (ODE and Basis)
		self.A = A
		self.mu = mu
		
		# ODE
		self.g = g # positive, default random normalized 
		self.dt = dt
		self.M = M

		# Basis
		self.LL = LL
		self.err = err

# Characterizing alg
class Algorithm:
	def __init__(self, outer=None, inner=None, Tmax=None, alpha=None, inner_g=None, lowRank=None, \
		Sparse=None, GroupSparse=None, storeLL=None, storeErr=None, rho=None, thres=None, \
		alphaLR=None, alphaS=None, alphaGS=None, truth=None):
		
		# both (ODE and Basis)
		self.outer = outer
		self.inner = inner
		self.Tmax = Tmax # list

		# ODE
		self.alpha = alpha
		self.inner_g = inner_g

		# Basis (how to test?)
		self.LowRank = LowRank
		self.Sparse = Sparse
		self.GroupSparse = GroupSparse
		self.storeLL = storeLL
		self.storeErr = storeErr
		self.rho = rho
		self.thres = thres
		self.alphaLR = alphaLR
		self.alphaS = alphaS
		self.alphaGS = alphaGS 
		self.truth = truth # also have alg.truth.A, alg.truth.mu

# Characterizing element of Seqs
class Sequence:
	def __init__(self, Start=None, Stop=None, Time=None, Mark=None):
		# both (ODE and Basis)
		self.Start = Start
		self.Stop = Stop
		self.Time = Time
		self.Mark = Mark

# Characterizing para, used in Simulation_Branch_HP to get Seqs
class Parameter:
	def __init__(self, kernel=None, w=None, landmark=None, mu=None, A=None):
		# Below are in MMHP_testing_ODE.py
		para.kernel = kernel 
		para.w = w 
		para.landmark = landmark
		para.mu = mu
		para.A = A

# Characterizing options, used in Simulation_Branch_HP to get Seqs
class Options:
	def __init__(self, N=None, Nmax=None, Tmax=None, tstep=None, dt=None, M=None, GenerationNum=None):
		# Below are in MMHP_testing_ODE.py
		self.N = N
		self.Nmax = Nmax
		self.Tmax = Tmax
		self.tstep = tstep
		self.dt = dt
		self.M = M
		self.GenerationNum = GenerationNum
