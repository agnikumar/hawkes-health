%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter studies: lambda_1, lambda_2, lambda_3, alpha, M vs. NLL
% Learning_MLE_Basis.m (lambda_1, lambda_2)
% Exp1_Learning_MLE_Basis.m (lambda_1, lambda_2, lambda_3)
% NLL expression does not involve parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Superposed Hawkes Processes: various source_num, D = 10
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

disp('Maximum likelihood estimation and basis representation')        

% simulation options 
options.N = 50; % 200; % the number of sequences
options.Nmax = 100; % the maximum number of events per sequence
options.Tmax = 50; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1;
options.M = 250;
options.GenerationNum = 10;
D = 1; %3; % the dimension of Hawkes processes
%nTest = 1;
%nSeg = 5;

% simulation parameters
% disp('Approximate simulation of Hawkes processes via branching process')
% disp('Complicated gaussian kernel')
para.kernel = 'gauss';
para.w = 1.5; 
para.landmark = 0:4:12;
L = length(para.landmark);
para.mu = rand(D,1)/D;
para.A = zeros(D, D, L);
for l = 1:L
    para.A(:,:,l) = (0.5^l)*(0.5+rand(D));
end
para.A = 0.9*para.A./max(abs(eig(sum(para.A,3))));
para.A = reshape(para.A, [D, L, D]);
Seqs = Simulation_Branch_HP(para, options);

alg.LowRank = 1; % on
alg.Sparse = 1; % on 
alg.alphaLR = 0.5; % grid search
alg.alphaS = 1; % grid search
alg.alphaLR_mu = 0.5; % added, grid search

alg.outer = 5; %5;
alg.rho = 0.1;
alg.inner = 8; %8;
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off
alg.storeErr = 0; % off
alg.storeLL = 0; % off 

% initialize (Basis)
model.A = rand(D,L,D)./(L*D^2);
model.mu = rand(D,1)./D;
model.kernel = 'gauss';
model.w = 2;
model.landmark = para.landmark;

% % initialize (ODE)
% alg.inner_g = 100;
% alg.alpha = 10000;
% model.M = 1000;
% model.D = L; %options.N; 
% model.dt = 0.02;
% model.g = rand(model.M, model.D);
% model.g = model.g./repmat(sum(model.g),[model.M,1]);

% toggle
model_Basis = model; % Basis representation (A, mu)
% model_ODE = model;
% model_type = "Basis";
%model_type = "ODE";

% if model_type == "Basis"
    %model_Basis = Initialization_Basis(Seqs);
    %model_Basis = Learning_MLE_Basis(Seqs, model, alg);
    %model_select = model_Basis;
    model_Basis = Exp1_Learning_MLE_Basis(Seqs, model_Basis, alg); 
    [A_Basis, Phi_Basis] = ImpactFunc(model_Basis, options);
    model_select = model_Basis;
% elseif model_type == "ODE"
%     times = model.dt:model.dt:(model.dt * model.M); % TODO: check
%     para.g = Kernel(times, para); % ground truth g, TODO: check
%     model_ODE = Learning_MLE_ODE(Seqs, model_ODE, alg);
%     [A_ODE, Phi_ODE] = ImpactFunc_ODE(model_ODE);
%     model_select = model_ODE;
%     g_error = norm(model_select.g - para.g)/norm(para.g); % with model_ODE
% end
    
A_error = norm(model_select.A - para.A)/norm(para.A);
mu_error = norm(model_select.mu - para.mu)/norm(para.mu);
