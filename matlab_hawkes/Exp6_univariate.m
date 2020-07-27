%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Univariate real data experiment: U = 41, C = 1, for 41 MMHP models
% Expect 41 (A, mu) pairs, and one global triggering kernel g
% A: 1 by 1 by 1 (41 A values)
% mu: 1 by 1 (41 mu values)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

D = 1;
U = 1; % one unit in each run of loop
%num_units = 41;
num_units = 20; %16; %30;

A_uni_vals = [];
mu_uni_vals = [];
LL_test_vals = [];

for x = 1:num_units % total number of units
    seqs_unit_train = struct();
    seqs_unit_test = struct();
    time_vect = load(sprintf('time_%d.mat', x));
    mark_vect = load(sprintf('mark_%d.mat', x));
    lx = (length(time_vect.time));
    %half = ceil(lx/2);
    half = lx; % train on all data
    
    seqs_unit_train(1).Time = time_vect.time(1:half);
    seqs_unit_test(1).Time = time_vect.time(half+1:end);
    
    seqs_unit_train(1).Mark = mark_vect.mark(1:half);
    seqs_unit_test(1).Mark = mark_vect.mark(half+1:end);
    
    seqs_unit_train(1).Start = 0;
    seqs_unit_test(1).Start = 0;
    
    seqs_unit_train(1).Stop = max(time_vect.time(1:half)); % maximum timestamp over particular unit
    seqs_unit_test(1).Stop = max(time_vect.time(half+1:end));
    
    seqs_unit_train(1).Feature = [];
    seqs_unit_test(1).Feature = [];
    
    %disp(seqs_unit);
    
    alg.LowRank = 0;
    alg.Sparse = 0;
    alg.GroupSparse = 0;
    alg.outer = 4; % perviously, 4
    alg.rho = 0.1;
    alg.inner = 20; % perviously, 5 or 6 % 60 good
    alg.thres = 1e-5;
    alg.Tmax = [];
    alg.storeErr = 0;
    alg.storeLL = 0;
    
    alg.updatemu = 1;
    
    model.kernel = 'gauss';
    model.w = 0.5; % originally, 2 % 1 good
    model.landmark = 100; %10; %0; %0:1:40; % 10 or 100 good, EMD goes down at 200 
    
    nTest = 5; %5 %50;
    nSeg = 1; %8;
    %nNum = floor((U*0.5)/nSeg);
    
    A_uni = zeros(1,nSeg,nTest);
    mu_uni = zeros(1,nSeg,nTest);
    LL = zeros(1,nSeg,nTest);
    for n = 1:nTest
        for i = 1:nSeg
            % initialize
            model.A = rand(D,U,D)./(U*D^2);
            model.mu = rand(D,U)./D;
            %disp(seqs_unit_train(1));
            output = Learning_MLE_Basis_MTmu(seqs_unit_train(1), model, alg);
            %LL(1,i,n) = Loglike_Basis(seqs_unit_test(1), output, alg); % test LL
            disp(output.A);
            A_uni(1,i,n) = output.A; % learned A
            mu_uni(1,i,n) = output.mu;  % learned mu
        end
    end 
%     LL_mean = mean(LL, 3); 
%     LL_std = std(LL, 0, 3);
%     LL_stats = [LL_mean; LL_std];
%     LL_test_vals = vertcat(LL_test_vals, LL_stats); 
    
    A_uni_mean = mean(A_uni, 3); 
    A_uni_std = std(A_uni, 0, 3);
    A_uni_stats = [A_uni_mean; A_uni_std];
    A_uni_vals = vertcat(A_uni_vals, A_uni_stats);
    
    mu_uni_mean = mean(mu_uni, 3); 
    mu_uni_std = std(mu_uni, 0, 3);
    mu_uni_stats = [mu_uni_mean; mu_uni_std];
    mu_uni_vals = vertcat(mu_uni_vals, mu_uni_stats);
    
    disp("Done with one unit.");
end

%data = [LL_test_vals; A_uni_vals; mu_uni_vals];
data = [A_uni_vals; mu_uni_vals];
%writematrix(data, "new_data/univariate_10runs_30units_largew_2.csv"); % mean rows followed by std rows (A, mu)
writematrix(data, "new_data/univariate_biglandmark.csv");