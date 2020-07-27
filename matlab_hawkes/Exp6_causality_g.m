%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combination of: 
% Exp1_Learning_MLE_Basis.m (lambda_1, lambda_2, lambda_3)
% Exp1_Learning_MLE_ODE.m (for getting g)
% NLL expression does not involve parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
     
% % simulation options 
% options.N = 20; % 200; % the number of sequences
% options.Nmax = 100; %100; % the maximum number of events per sequence
% options.Tmax = 50; % the maximum size of time window
% options.tstep = 0.1;
% options.dt = 0.1;
% options.M = 250;
% options.GenerationNum = 10;
% D = 1; %3; % the dimension of Hawkes processes (number of classes)
% %nTest = 1;
% %nSeg = 5;
% 
% % simulation parameters
% % disp('Approximate simulation of Hawkes processes via branching process')
% % disp('Complicated gaussian kernel')
% para.kernel = 'gauss';
% para.w = 1.5; 
% para.landmark = 0:1:19; %0:4:12;
% L = length(para.landmark); % should be the same as number of units (?)
% para.mu = rand(D,1)/D;
% para.A = zeros(D, D, L);
% for l = 1:L
%     para.A(:,:,l) = (0.5^l)*(0.5+rand(D));
% end
% para.A = 0.9*para.A./max(abs(eig(sum(para.A,3))));
% para.A = reshape(para.A, [D, L, D]);
% Seqs = Simulation_Branch_HP(para, options);

D = 20;
U = 1;
L = U;

time_vect = load('time_all.mat'); % top 20 units time_all_lots
mark_vect = load('mark_all.mat');

seqs_big(1).Time = time_vect.time_all; % .time_all
seqs_big(1).Mark = mark_vect.mark_all; % .mark_all

seqs_big(1).Start = 0;
seqs_big(1).Stop = 545; % maximum timestamp over all units
seqs_big(1).Feature = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alg.LowRank = 1; % on
alg.Sparse = 1; % on
alg.alphaLR = 0.1; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.001; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.001; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

alg.outer = 3; % number of runs
alg.rho = 0.1;
alg.inner = 5; % number of loops (as related to convergence)
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off
alg.storeErr = 0; % on
alg.storeLL = 0; % on 

% initialize (Basis)
model.A = rand(D,L,D)./(L*D^2);
model.mu = rand(D,1)./D;
model.kernel = 'gauss';
model.w = 2;
model.landmark = 0:1:19;

% initialize (ODE)
alg.inner_g = 100;
alg.alpha = 100; %10000;
model.M = 1000; %1000;
model.D = 20; %options.N; 
model.dt = 0.02;
model.g = rand(model.M, model.D); % global triggering kernel of length M
model.g = model.g./repmat(sum(model.g),[model.M,1]);

% for error calculations 
model.truth = model; % added
model.truth.g = model.g; % added

model_both = model;
model_both = Exp1_Basis_ODE_variant(seqs_big, model_both, alg);
%model_both = Learning_MLE_ODE(seqs_big, model_both, alg);

[A_both, Phi_both] = ImpactFunc_ODE(model_both);

disp(model_both.g)


