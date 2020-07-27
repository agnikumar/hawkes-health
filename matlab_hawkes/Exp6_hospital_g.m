%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real data: U = 41, C = 1
% Expect 41 (A, mu) pairs, and one global triggering kernel g
% A: 1 by 1 by 41
% mu: 1 by 41
% Looking at how LL changes are more data is included in training set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

D = 1;
U = 1;

% A_uni_vals = [];
% mu_uni_vals = [];
% LL_test_vals = [];
% 
% seqs_all = struct();
% for n = 1:U % total number of units
%     time_vect = load(sprintf('time_%d.mat', n));
%     mark_vect = load(sprintf('mark_%d.mat', n));
%     seqs_all(n).Time = time_vect.time;
%     seqs_all(n).Mark = mark_vect.mark;
%     seqs_all(n).Start = 0;
%     seqs_all(n).Stop = 545; % maximum timestamp over all units
%     seqs_all(n).Feature = [];
% end

seqs_big = struct();
time_vect = load('time_all.mat'); % top 20 units time_all_lots
%mark_vect = load('mark_all.mat');

seqs_big(1).Time = time_vect.time_all; % .time_all
%seqs_big(1).Mark = mark_vect.mark_all; % .mark_all
seqs_big(1).Mark = ones(1,1033);

seqs_big(1).Start = 0;
seqs_big(1).Stop = 545; % maximum timestamp over all units
seqs_big(1).Feature = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alg.LowRank = 0;
alg.Sparse = 0;
alg.GroupSparse = 0;
alg.alphaLR = 0.2; %0.2; %0.2; %0.1; % grid search, lambda_1, between 0 and 0.2
alg.alphaS = 0.004; %0.004; %0.001; %1; % grid search, lambda_2, between 0 and 0.002
alg.alphaLR_mu = 0.002; %0.004; %0.004; %0.001; %0.5; % added, grid search, lambda_3, between 0 and 0.002

alg.outer = 5; % perviously, 4
alg.rho = 0.1;
alg.inner = 8; %5 % perviously, 15
alg.thres = 1e-5;
alg.Tmax = [];
alg.storeErr = 0;
alg.storeLL = 0;

alg.updatemu = 1;

model.kernel = 'gauss';
model.w = 1; %0.002; %2;
model.landmark = 0; %0:0.025:1; %rand(1, U); %0:1:40; %0:3:2; %0:1:40; %0;
%model.kernel = 'exp';
%model.w = 1;
%model.landmark = 0;

nTest = 10; %5
nSeg = 1; % number of chunks of data
%nNum = U/nSeg;
nNum = floor((U*0.5)/nSeg);
%nNum = floor(U/nSeg);

%A_multi = zeros(1,nSeg,nTest);
%mu_multi = zeros(1,nSeg,nTest);
A_multi = [];
mu_multi = [];

Phi_list = [];

LL = zeros(1,nSeg,nTest);
for n = 1:nTest
    for i = 1:nSeg
        model.A = rand(D,U,D)./(U*D^2);
        model.mu = rand(D,U)./D;
        
        alg.inner_g = 100; % originally 100
        alg.alpha = 1000; %10000;
        model.M = 550; %1000; 540 days
        model.D = U; %U; %options.N; 
        model.dt = 1; %0.1; %1; %0.02;
        model.g = rand(model.M, model.D); % global triggering kernel of length M
        model.g = model.g./repmat(sum(model.g),[model.M,1]);
        
        %output = Learning_MLE_Basis_MTmu(seqs_all(1:end), model, alg);
        output = Exp1_Basis_ODE_variant(seqs_big(1), model, alg); % seqs_all(1)
        
        %A_multi(1,i,n) = output.A; % learned A
        %mu_multi(1,i,n) = output.mu; % learned mu
        %A_multi = vertcat(A_multi, output.A);
        %mu_multi = vertcat(mu_multi, output.mu);
        
        %LL(1,i,n) = Loglike_Basis(seqs_all(floor(U*0.5)+1:U), output, alg);
        %LL(1,i,n) = Loglike_Basis(seqs_all(1:i*nNum), output, alg);
        %LL(1,i,n) = Loglike_Basis(seqs_all(1), output, alg);
        
        [A_both, Phi_both] = ImpactFunc_ODE(output);
        Phi_list = vertcat(Phi_list, Phi_both);
        
        
    end
    disp("Done with a test run.");
end

% LL_mean = mean(LL, 3); 
% LL_std = std(LL, 0, 3);
% LL_stats = [LL_mean; LL_std];
% 
% %A_multi_mean = mean(A_uni, 3); 
% %A_multi_std = std(A_uni, 0, 3);
% A_multi_mean = mean(A_multi);
% A_multi_std = std(A_multi);
% A_multi_stats = [A_multi_mean; A_multi_std];
% 
% %mu_multi_mean = mean(mu_multi, 3); 
% %mu_multi_std = std(mu_multi, 0, 3);
% mu_multi_mean = mean(mu_multi);
% mu_multi_std = std(mu_multi);
% mu_multi_stats = [mu_multi_mean; mu_multi_std];
% 
% writematrix(LL_stats, "data/multi_LL_trial_4.csv"); 
% writematrix(A_multi_stats, "data/multi_A_trial_4.csv"); 
% writematrix(mu_multi_stats, "data/multi_mu_trial_4.csv"); 

%[A_both, Phi_both] = ImpactFunc_ODE(output);

Phi_mean = mean(Phi_list);

writematrix(Phi_list, "new_data/multi_g_smalldt.csv");
%disp(output.g);
