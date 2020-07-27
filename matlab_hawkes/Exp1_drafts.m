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
options.N = 20; % 200; % the number of sequences
options.Nmax = 100; % the maximum number of events per sequence
options.Tmax = 50; % the maximum size of time window
options.tstep = 0.1;
options.dt = 0.1;
options.M = 250;
options.GenerationNum = 10;
D = 1; %3; % the dimension of Hawkes processes (number of classes)
%nTest = 1;
%nSeg = 5;

% simulation parameters
% disp('Approximate simulation of Hawkes processes via branching process')
% disp('Complicated gaussian kernel')
para.kernel = 'gauss';
para.w = 1.5; 
para.landmark = 0:4:12;
L = length(para.landmark); % should be the same as number of units (?)
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

alg.outer = 1; %5;
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

% initialize (ODE)
alg.inner_g = 100;
alg.alpha = 10000;
model.M = 1000;
model.D = L; %options.N; 
model.dt = 0.02;
model.g = rand(model.M, model.D); % global triggering kernel of length M
model.g = model.g./repmat(sum(model.g),[model.M,1]);

model_Basis = model; % Basis representation (A, mu)
model_ODE = model;
% toggle
%model_type = "Basis";
model_type = "ODE";

if model_type == "Basis"
    %model_Basis = Initialization_Basis(Seqs);
    %model_Basis = Learning_MLE_Basis(Seqs, model, alg);
    model_Basis = Exp1_Learning_MLE_Basis(Seqs, model_Basis, alg); 
    [A_Basis, Phi_Basis] = ImpactFunc(model_Basis, options);
    model_select = model_Basis;
elseif model_type == "ODE"
    %times = model.dt:model.dt:(model.dt * model.M); % TODO: check
    %para.g = Kernel(times, para); % ground truth g, TODO: check
    [~, para.g] = ImpactFunc_ODE(model);
    model_ODE = Learning_MLE_ODE(Seqs, model_ODE, alg);
    [A_ODE, Phi_ODE] = ImpactFunc_ODE(model_ODE);
    model_select = model_ODE;
    %g_error = norm(model_select.g - para.g)/norm(para.g); % with model_ODE
    % g_error = norm(Phi_ODE - para.g)/norm(para.g);
    
    %fun = @(a_est, g_est, a_orig, g_orig) (a_est*g_est - a_orig*g_orig).^2;
    %g_error = (1/(L*D^2)) * integral(fun(model_select.A, Phi_ODE, para.A, para.g), 0, Inf);
    %model_ODE.g = Phi_ODE;
    %model.g = para.g;
    
    %model_ODE.g = reshape(model_ODE.g, [model.M, 1]);
    %model.g = reshape(model.g, [model.M, 1]);
    
    model_ODE.g = reshape(Phi_ODE, [model.M, 1]); % yields very small error
    model.g = reshape(para.g, [model.M, 1]); % yields very small error
    
    model_new = model;
    model_new.g = (model_ODE.g - model.g).^2; % most correct
    model_true = model;
    model_true.g = (model.g).^2;
    %model_new.g = (sum(model_ODE.A*model_ODE.g) - sum(model.A*model.g)).^2;
%     g_error = (1/(L*D^2)) * (Kernel_Integration_Approx(model.dt, model_ODE) - ...
%                 Kernel_Integration_Approx(model.dt, model));
    % g_error = norm(model_ODE.g - model.g)/norm(model.g); % high error
    %g_error = (1/(L*D^2)) * (Kernel_Integration_Approx(model.dt, (model_ODE.g - model.g).^2));
    L = options.N;
    %g_error = (1/(L*D^2)) * (Kernel_Integration_Approx(model.dt, model_new));
    %g_error = (1/(L*D^2)) * (Kernel_Integration_Approx(model.dt, model_new));
    g_error = (Kernel_Integration_Approx(model.dt, model_new))/(Kernel_Integration_Approx(model.dt, model_true));
end
    
A_error = norm(model_select.A - para.A)/norm(para.A);
mu_error = norm(model_select.mu - para.mu)/norm(para.mu);
