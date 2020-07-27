%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Testing various upper bounds for A, mu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

options.N = 50; % the number of sequences
options.Nmax = 100; % the maximum number of events per sequence
options.Tmax = 100; % the maximum size of time window
options.tstep = 0.2;% the step length for computing sup intensity
options.M = 50; % the number of steps
options.GenerationNum = 5; % the number of generations
D = 4; % the dimension of Hawkes processes
nTest = 10;
nSeg = 1; % only observing final error (5 originally)
nNum = options.N/nSeg;

% disp('Fast simulation of Hawkes processes with exponential kernel')
% para1.mu = rand(D,1)/D;
% para1.A = rand(D, D);
% para1.A = 0.25 * para1.A./max(abs(eig(para1.A)));
% para1.A = reshape(para1.A, [D, 1, D]);
% para1.w = 1;
% Seqs1 = Simulation_Branch_HP(para1, options);
% Seqs1 = SimulationFast_Thinning_ExpHP(para1, options);

% disp('Thinning-based simulation of Hawkes processes with exponential kernel')
% para2 = para1;
% para2.kernel = 'exp';
% para2.landmark = 0;
% Seqs2 = Simulation_Thinning_HP(para2, options);
% 
% disp('Approximate simulation of Hawkes processes via branching process')
% para3 = para2;
% Seqs3 = Simulation_Branch_HP(para3, options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UB_1 = 1;
para1.mu = unifrnd(0, UB_1, [D, 1])/D; % with UB
para1.A = unifrnd(0, UB_1, [D, D]); % with UB

para1.A = para1.A./max(abs(eig(para1.A)));
para1.A = 0.25 * reshape(para1.A, [D, 1, D]);
para1.w = 1;
para1.kernel = 'exp';
para1.landmark = 0;
Seqs1 = Simulation_Branch_HP(para1, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UB_2 = 2;
para2.mu = unifrnd(0, UB_2, [D, 1])/D; % with UB
para2.A = unifrnd(0, UB_2, [D, D]); % with UB

para2.A = 0.25 * para2.A./max(abs(eig(para2.A)));
para2.A = reshape(para2.A, [D, 1, D]);
para2.w = 1;
para2.kernel = 'exp';
para2.landmark = 0;
Seqs2 = Simulation_Branch_HP(para2, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UB_4 = 4;
para4.mu = unifrnd(0, UB_4, [D, 1])/D; % with UB
para4.A = unifrnd(0, UB_4, [D, D]); % with UB

para4.A = 0.25 * para4.A./max(abs(eig(para4.A)));
para4.A = reshape(para4.A, [D, 1, D]);
para4.w = 1;
para4.kernel = 'exp';
para4.landmark = 0;
Seqs4 = Simulation_Branch_HP(para4, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UB_8 = 8;
para8.mu = unifrnd(0, UB_8, [D, 1])/D; % with UB
para8.A = unifrnd(0, UB_8, [D, D]); % with UB

para8.A = 0.25 * para8.A./max(abs(eig(para8.A)));
para8.A = reshape(para8.A, [D, 1, D]);
para8.w = 1;
para8.kernel = 'exp';
para8.landmark = 0;
Seqs8 = Simulation_Branch_HP(para8, options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Learning Hawkes processes from synthetic data')
alg.LowRank = 0;
alg.Sparse = 0;
alg.GroupSparse = 0;
alg.outer = 4; % perviously, 4
alg.rho = 0.1;
alg.inner = 5; % perviously, 5
alg.thres = 1e-5;
alg.Tmax = [];
alg.storeErr = 0;
alg.storeLL = 0;

disp('Evaluation of quality of synthetic data.')

%Err = zeros(3,nSeg,nTest);
Err = zeros(4,nSeg,nTest);
for n = 1:nTest
    for i = 1:nSeg
        % initialize
        model.A = rand(D,1,D)./(D^2);
        model.mu = rand(D,1)./D;
        model.kernel = 'exp';
        model.w = 1;
        model.landmark = 0;

%         model1 = model;
%         model2 = model;
%         model3 = model;
%         model1 = Learning_MLE_Basis( Seqs1(1:i*nNum), model1, alg );
%         model2 = Learning_MLE_Basis( Seqs2(1:i*nNum), model2, alg );
%         model3 = Learning_MLE_Basis( Seqs3(1:i*nNum), model3, alg );
        model1 = model;
        model2 = model;
        model4 = model;
        model8 = model;
        model1 = Learning_MLE_Basis(Seqs1(1:i*nNum), model1, alg);
        model2 = Learning_MLE_Basis(Seqs2(1:i*nNum), model2, alg);
        model4 = Learning_MLE_Basis(Seqs4(1:i*nNum), model4, alg);
        model8 = Learning_MLE_Basis(Seqs8(1:i*nNum), model8, alg);
        disp("Done with one test round.");

%         Err(1,i,n) = norm([model1.mu; model1.A(:)] - [para1.mu; para1.A(:)])/...
%             norm([para1.mu; para1.A(:)]);
%         Err(2,i,n) = norm([model2.mu; model2.A(:)] - [para2.mu; para2.A(:)])/...
%             norm([para2.mu; para2.A(:)]);
%         Err(3,i,n) = norm([model3.mu; model3.A(:)] - [para3.mu; para3.A(:)])/...
%             norm([para3.mu; para3.A(:)]);
        Err(1,i,n) = norm([model1.mu; model1.A(:)] - [para1.mu; para1.A(:)])/...
            norm([para1.mu; para1.A(:)]);
        Err(2,i,n) = norm([model2.mu; model2.A(:)] - [para2.mu; para2.A(:)])/...
            norm([para2.mu; para2.A(:)]);
        Err(3,i,n) = norm([model4.mu; model4.A(:)] - [para4.mu; para4.A(:)])/...
            norm([para4.mu; para4.A(:)]);
        Err(4,i,n) = norm([model8.mu; model8.A(:)] - [para8.mu; para8.A(:)])/...
            norm([para8.mu; para8.A(:)]);
    end
end

Error = mean(Err, 3); 
Std = std(Err, 0, 3);
figure
hold on
for i = 1:4
    errorbar(nNum:nNum:options.N, Error(i,:), Std(i,:), 'o-');
end
hold off
axis tight
xlabel('The number of training sequences');
ylabel('Relative estimation error')
legend('UB = 1', 'UB = 2', 'UB = 4', 'UB = 8')
title('Learning results based on different simulation methods')

stats = [Error; Std];
writematrix(stats, "data/upper_bounds_newer.csv"); % mean rows followed by std rows
