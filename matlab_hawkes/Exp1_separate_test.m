%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter studies: lambda_1, lambda_2, lambda_3, alpha, M vs. NLL
% Learning_MLE_Basis.m (lambda_1, lambda_2)
% Exp1_Learning_MLE_Basis.m (lambda_1, lambda_2, lambda_3)
% NLL expression does not involve parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
     
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
alg.alphaLR = 0.5; % grid search, lambda_1
alg.alphaS = 1; % grid search, lambda_2
alg.alphaLR_mu = 0.5; % added, grid search, lambda_3

alg.outer = 1; % number of runs
alg.rho = 0.1;
alg.inner = 10; % number of loops (as related to cnvergence)
alg.thres = 1e-5;
alg.Tmax = [];
alg.GroupSparse = 0; % off

alg.storeErr = 1; % on
alg.storeLL = 1; % on 
alg.truth.A = para.A; % added
alg.truth.mu = para.mu; % added

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

% for error calculations (ODE)
model.truth = model;
model.truth.g = model.g;

model_Basis = model; % Basis representation (A, mu)
model_ODE = model;
% toggle
%model_type = "Basis";
model_type = "ODE";

model_type_Amu = "Basis"; % use Basis for A and mu error calculations
% model_type_Amu = "ODE"; % use ODE for A and mu error calculations

if model_type == "Basis" || model_type_Amu == "Basis" % should always be executed
    %model_Basis = Initialization_Basis(Seqs);
    %model_Basis = Learning_MLE_Basis(Seqs, model, alg);
    model_Basis = Exp1_Learning_MLE_Basis(Seqs, model_Basis, alg); 
    [A_Basis, Phi_Basis] = ImpactFunc(model_Basis, options);
    % model_select = model_Basis;
end

if model_type == "ODE" || model_type_Amu == "ODE"
    model_ODE = Exp1_Learning_MLE_ODE(Seqs, model_ODE, alg);
    [A_ODE, Phi_ODE] = ImpactFunc_ODE(model_ODE);
    % model_select = model_ODE; 
    model_ODE_eval = model;
    model_ODE_eval.g = (model_ODE.g - model.g).^2; % most correct
    model_eval = model;
    model_eval.g = (model.g).^2;
    % g_error = norm(model_ODE.g - model.g)/norm(model.g);
    g_error = (Kernel_Integration_Approx(model.dt, model_ODE_eval)) / ...
        (Kernel_Integration_Approx(model.dt, model_eval));
end

model_select = model_Basis;
A_error = norm(model_select.A - para.A)/norm(para.A);
mu_error = norm(model_select.mu - para.mu)/norm(para.mu);

disp("A_error (Basis): " + A_error); % last round (o)
disp("mu_error (Basis): " + mu_error); % last round (o)
disp("g_error (ODE): " + g_error);

disp("Basis errors:");
disp(size(model_select.err));
disp(model_select.err); % order of error columns: mu, A, both
disp("Basis log-likelihood (not NLL):");
disp(size(model_select.LL));
disp(model_select.LL);

disp("ODE errors:");
disp(size(model_ODE.err));
disp(model_ODE.err);
disp("ODE log-likelihood (not NLL):");
disp(size(model_ODE.LL));
disp(model_ODE.LL);

