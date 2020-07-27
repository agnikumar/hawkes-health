%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constructing infectivity matrix
% Hospital level analysis
% Real data: U = 1, C = 41
% A: 41-by-41 infectivity matrix, 
% mu: 41-by-1 intensity vector
% g: 1-by-1 
% A: 1 by 1 by 41
% mu: 1 by 41
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

D = 20; 
%D = 41;

U = 1;

options.N = 20; % the number of sequences
options.Nmax = 85; % the maximum number of events per sequence
options.Tmax = 545; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1; % the length of each time step
options.M = 500; % the number of steps in the time interval for computing sup-intensity
options.GenerationNum = 100; % the number of generations for branch processing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading in medical data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% all
time_vect = load('time_all_lots.mat'); % top 20 units
mark_vect = load('mark_all_lots.mat');
% time_vect = load('time_all.mat'); 
% mark_vect = load('mark_all.mat');

seqs_big(1).Time = time_vect.time; % .time_all
seqs_big(1).Mark = mark_vect.mark; % .mark_all
% seqs_big(1).Time = time_vect.time_all; 
% seqs_big(1).Mark = mark_vect.mark_all;

seqs_big(1).Start = 0;
seqs_big(1).Stop = 545; % maximum timestamp over all units
seqs_big(1).Feature = [];

%[A, Phi] = ImpactFunc(para1, options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alg.LowRank = 0; % without low-rank regularizer
alg.Sparse = 1; % with sparse regularizer
alg.alphaS = 1;
alg.GroupSparse = 1; % with group-sparse regularizer
alg.alphaGS = 100;
alg.outer = 8;
alg.rho = 0.1; % the initial parameter for ADMM
alg.inner = 5; %5;
alg.thres = 1e-5;
alg.Tmax = [];
alg.storeErr = 0;
alg.storeLL = 0;

%model = Initialization_Basis(seqs_big); 
% learning the model by MLE
%model = Learning_MLE_Basis(seqs_big, model, alg); 

model.kernel = 'gauss';
model.w = 0.5; %0.002; %2;
model.landmark = 0;

alg.alpha = 1000;
model.A = rand(D,U,D)./(U*D^2);
model.mu = rand(D,1)./D;
model.M = 500;
model.D = U;
model.dt = 0.02;
model.g = rand(model.M, model.D); % global triggering kernel of length M
model.g = model.g./repmat(sum(model.g),[model.M,1]);

model = Exp1_Basis_ODE_variant(seqs_big, model, alg); % for getting g %Exp1_Basis_ODE_variant
[A1, Phi1] = ImpactFunc(model, options);

% Visualize the infectivity matrix (the adjacent matrix of Granger causality graph)
figure
subplot(121)        
imagesc(A1) % should be A, but no ground truth for real data
title('Ground truth of infectivity')
axis square
colorbar
subplot(122)        
imagesc(A1)
title('Estimated infectivity-MLE')
colorbar
axis square

%writematrix(A1, "new_data/A_causality_new.csv")