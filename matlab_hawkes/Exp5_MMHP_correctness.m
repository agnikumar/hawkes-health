clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get 41-unit sequence structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = 1; % only 1 type of event, positive diagnosis
U = 41;
seqs_all_fresh = struct();
for n = 1:U % total number of units
    time_vect = load(sprintf('time_%d.mat', n));
    mark_vect = load(sprintf('mark_%d.mat', n));
    seqs_all_fresh(n).Time = time_vect.time;
    seqs_all_fresh(n).Mark = mark_vect.mark;
    seqs_all_fresh(n).Start = 0;
    seqs_all_fresh(n).Stop = 545; % maximum timestamp over all units
    seqs_all_fresh(n).Feature = [];
end

alg.LowRank = 0;
alg.Sparse = 0;
alg.GroupSparse = 0;
alg.storeLL = 0;
alg.storeErr = 0;
alg.updatemu = 1;
alg.thres = 1e-5;
alg.rho = 0.1;
alg.Tmax = [];

alg.outer = 1;
alg.inner = 20;

%model.kernel = 'gauss';
%model.w = 0.01; %0.002; 2;
%model.landmark = 0:3:9 ; %rand(1,41); %0; %zeros(1,41); %0; %0:10:400; %0; %0:0.25:10; % larger yields larger learned A

%model.A = rand(D,U,D)./(U*D^2);
%model.mu = rand(D,U)./D;
model = Initialization_Basis(seqs_all_fresh);

%output_fresh = Learning_MLE_Basis_MTmu(seqs_all_fresh(1:41), model, alg);
%LL_fresh = Loglike_Basis(seqs_all_fresh(1), output_fresh, alg);
output_fresh = Learning_MLE_Basis(seqs_all_fresh(1:20), model, alg);
LL_fresh = Loglike_Basis(seqs_all_fresh(21:40), output_fresh, alg);

%LL_fresh_official = Loglike_Basis(seqs_all(1), output_fresh, alg);

disp(output_fresh.A);
disp("------------------------");
disp(output_fresh.mu);
disp(LL_fresh);
%disp(LL_fresh_official);
