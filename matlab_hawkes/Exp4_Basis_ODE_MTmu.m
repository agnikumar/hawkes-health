function model = Exp4_Basis_ODE_MTmu(Seqs, model, alg)
                                                        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Adding lambda_3 (low-rank regularization on mu) to Learning_MLE_Basis
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial 
Aest = model.A;  
muest = model.mu;
gest = model.g; % added

%%%%%%%%%%%%%%%%%%%%%%%%%
ODE_Aest = model.A; 
ODE_muest = model.mu; 
%%%%%%%%%%%%%%%%%%%%%%%%%

if alg.LowRank
    UL = zeros(size(Aest));
    ZL = Aest;
    UL_mu = zeros(size(muest)); 
    ZL_mu = muest; 
end

if alg.Sparse
    US = zeros(size(Aest));
    ZS = Aest;
end

if alg.GroupSparse
    UG = zeros(size(Aest));
    ZG = Aest;
end

D = size(Aest, 1);

if alg.storeLL
    model.LL = zeros(alg.outer, 1);
end
if alg.storeErr
    %model.err = zeros(alg.outer, 4);
    model.err = zeros(alg.outer, 7);
end

tic;
for o = 1:alg.outer
    
    disp("------------------------------"); % segment divider 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    DM = zeros(size(model.g)); 
    CM = DM; 
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    rho = alg.rho * (1.1^o);
    
    for n = 1:alg.inner
        
        NLL = 0; % negative log-likelihood
        
        %Amu = zeros(D, 1);
        Amu = zeros(size(muest)); % CHANGED
        Bmu = Amu;
        %Cmu = Amu; % ADDED
        
        %%%%%%%%%%%%%%%%%%%%%%%%%
        ODE_Amu = zeros(D, 1);
        ODE_Bmu = ODE_Amu;
        %%%%%%%%%%%%%%%%%%%%%%%%%
        
        CmatA = zeros(size(Aest));
        AmatA = CmatA;
        BmatA = CmatA;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ODE_BmatA = zeros(size(ODE_Aest));
        ODE_AmatA = ODE_BmatA;
        ODE_AmatA = ODE_AmatA + 2 * alg.alpha * ODE_Aest;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if alg.LowRank
            %BmatA = BmatA + rho*(UL-ZL) + rho*(UL_mu-ZL_mu); 
            BmatA = BmatA + rho*(UL-ZL);
            AmatA = AmatA + rho;
            Bmu = Bmu + rho*(UL_mu-ZL_mu); % ADDED
            Amu = Amu + rho; % ADDED
        end
        if alg.Sparse
            BmatA = BmatA + rho*(US-ZS);
            AmatA = AmatA + rho;
        end
        if alg.GroupSparse
            BmatA = BmatA + rho*(UG-ZG);
            AmatA = AmatA + rho;
        end
        
        % E-step: evaluate the responsibility using the current parameters    
        for c = 1:length(Seqs)
            if ~isempty(Seqs(c).Time)
                Time = Seqs(c).Time;
                Event = Seqs(c).Mark;
                Tstart = Seqs(c).Start;

                if isempty(alg.Tmax)
                    Tstop = Seqs(c).Stop;
                else
                    Tstop = alg.Tmax;
                    indt = Time < alg.Tmax;
                    Time = Time(indt);
                    Event = Event(indt);
                end
                
                %Bmu(:,c) = Bmu(:,c) + Tstop - Tstart; % ADDED
                
                %Amu = Amu + Tstop - Tstart;
                Amu(:,c) = Amu(:,c) + Tstop - Tstart; % CHANGED
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %ODE_Amu = ODE_Amu + Tstop - Tstart;
                ODE_Amu(:,c) = ODE_Amu(:,c) + Tstop - Tstart; % CHANGED
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                dT = Tstop - Time;
                GK = Kernel_Integration(dT, model);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ODE_GK = Kernel_Integration_Approx(dT, model);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                Nc = length(Time);

                for i = 1:Nc

                    ui = Event(i);
                    %disp(ui);
                    BmatA(ui,:,:) = BmatA(ui,:,:)+...
                        double(Aest(ui,:,:)>0).*repmat(GK(i,:), [1,1,D]);
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    ODE_AmatA(ui,:,:) = ODE_AmatA(ui,:,:)+...
                        double(ODE_Aest(ui,:,:)>0).*repmat(ODE_GK(i,:), [1,1,D]);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    ti = Time(i);  
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if n == alg.inner
                        Nums = min([ceil(dT(i)/model.dt), size(CM,1)]);
                        CM(1:Nums,:) = CM(1:Nums,:) + ...
                            repmat(sum(ODE_Aest(ui,:,:), 3), [Nums,1]);
                    end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    %lambdai = muest(ui);
                    %pii = muest(ui);
                    lambdai = muest(ui,c); % CHANGED
                    pii = muest(ui,c); % CHANGED
                    pij = [];
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %ODE_lambdai = ODE_muest(ui);
                    %ODE_pii = ODE_muest(ui);
                    ODE_lambdai = ODE_muest(ui,c); % CHANGED
                    ODE_pii = ODE_muest(ui,c); % CHANGED
                    ODE_pij = [];
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if i>1

                        tj = Time(1:i-1);
                        uj = Event(1:i-1);
                        
                        dt = ti - tj;
                        %disp("----------------");
                        %disp(Time);
                        %disp("----------------");
                        %disp(i);
                        %disp(Time(1:10));
                        %disp(ti);
                        %disp(tj);
                        %disp(dt);
                        %disp(i);
                        
                        gij = Kernel(dt, model);
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        ODE_gij = Kernel_Approx(dt, model);
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            
                        auiuj = Aest(uj, :, ui);
                        pij = auiuj .* gij;
                        lambdai = lambdai + sum(pij(:));
                        
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        ODE_auiuj = ODE_Aest(uj, :, ui);
                        ODE_pij = ODE_auiuj .* ODE_gij;
                        ODE_lambdai = ODE_lambdai + sum(ODE_pij(:));
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                    end

                    NLL = NLL - log(lambdai);
                    pii = pii./lambdai;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    ODE_pii = ODE_pii./ODE_lambdai; 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if i>1
                        pij = pij./lambdai;
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        ODE_pij = ODE_pij./ODE_lambdai;
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                        if ~isempty(pij) && sum(pij(:))>0
                            for j = 1:length(uj)
                                uuj = uj(j);
                                CmatA(uuj,:,ui) = CmatA(uuj,:,ui) - pij(j,:); 
                                
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                ODE_BmatA(uuj,:,ui) = ODE_BmatA(uuj,:,ui) ...
                                    + ODE_Aest(uuj,:,ui) .* ODE_pij(j,:);

                                if n == alg.inner
                                    Nums = min([ceil(dt(j)/model.dt), size(DM,1)]);
                                    DM(Nums,:) = DM(Nums,:) + ODE_pij(j,:);
                                end
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            end
                        end
                    end

                    %Bmu(ui) = Bmu(ui) + pii;
                    Bmu(ui,c) = Bmu(ui,c) + pii; % CHANGED
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %ODE_Bmu(ui) = ODE_Bmu(ui) + ODE_pii; 
                    ODE_Bmu(ui,c) = ODE_Bmu(ui,c) + ODE_pii; % CHANGED
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end

                %NLL = NLL + (Tstop-Tstart).*sum(muest);
                NLL = NLL + (Tstop-Tstart).*sum(muest,c); % CHANGED
                NLL = NLL + sum( sum( GK.*sum(Aest(Event,:,:),3) ) );
                %NLL = NLL + sum( sum( GK(c).intG.*sum(Aest(Event,:,:),3) ) );

            else
                warning('Sequence %d is empty!', c)
            end
        end
                
        % M-step: update parameters
        mu = Bmu./Amu;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        ODE_mu = ODE_Bmu./ODE_Amu;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if alg.Sparse==0 && alg.GroupSparse==0 && alg.LowRank==0
            A = -CmatA./BmatA; %(-BA+sqrt(BA.^2-4*AA*CA))./(2*AA);
            A(isnan(A))=0;
            A(isinf(A))=0;
        else            
            A = (-BmatA + sqrt(BmatA.^2 - 4*AmatA.*CmatA))./(2*AmatA);
            A(isnan(A))=0;
            A(isinf(A))=0;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ODE_A = abs(sqrt(ODE_BmatA./ODE_AmatA));
        ODE_A(isnan(ODE_A))=0;
        ODE_A(isinf(ODE_A))=0;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % check convergence
        Err=sum(sum(sum(abs(A-Aest))))/sum(abs(Aest(:)));
        Err_A = norm(A(:) - Aest(:))/norm(Aest(:)); % added
        Err_mu = norm(mu(:) - muest(:))/norm(muest(:)); % added
        %Err_g = norm(model.g(:) - gest(:))/norm(gest(:)); % added
        
        Aest = A;
        muest = mu;
        %%%%%%%%%%%%%%%%%%%
        ODE_Aest = ODE_A; 
        ODE_muest = ODE_mu; 
        %%%%%%%%%%%%%%%%%%%
        model.A = Aest;
        model.mu = muest;
%         fprintf('Outer=%d, Inner=%d, Obj=%f, RelErr=%f, Time=%0.2fsec\n', ...
%                o, n, NLL, Err, toc);
            
        if Err<alg.thres || (o==alg.outer && n==alg.inner)
            break;
        end    
    end
    
    if alg.LowRank
        threshold = alg.alphaLR/rho;
        ZL = SoftThreshold_LR( Aest+UL, threshold );
        UL = UL + (Aest-ZL);
        threshold_mu = alg.alphaLR_mu/rho; 
        ZL_mu = SoftThreshold_LR(Aest+UL_mu, threshold_mu); 
        UL_mu = UL_mu + (Aest-ZL_mu); 
    end
    
    if alg.Sparse
        threshold = alg.alphaS/rho;
        ZS = SoftThreshold_S( Aest+US, threshold );
        US = US + (Aest-ZS);
    end

    if alg.GroupSparse
        threshold = alg.alphaGS/rho;
        ZG = SoftThreshold_GS( Aest+UG, threshold );
        UG = UG + (Aest-ZG);
    end   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DM = DM./model.dt;
    CM = CM./model.g;
    
    gest = model.g;

    for n = 1:alg.inner_g
        for m = 1:size(model.g, 1)
            switch m
                case 1
                    a = 2*alg.alpha + CM(m,:) * (model.dt^2);
                    b = -2*alg.alpha*model.g(m+1,:);
                    c = -DM(m,:) * (model.dt^2);
                case size(model.g, 1)
                    a = 4*alg.alpha + CM(m,:) * (model.dt^2);
                    b = -2*alg.alpha*model.g(m-1,:);
                    c = -DM(m,:) * (model.dt^2);
                otherwise
                    a = 4*alg.alpha + CM(m,:) * (model.dt^2);
                    b = -2*alg.alpha * (model.g(m-1,:)+model.g(m+1,:));
                    c = -DM(m,:) * (model.dt^2);
            end
            
            gest(m,:) = (-b+sqrt(b.^2 - 4.*a.*c))./(2*a);
        end

        model.g = abs(gest); 
        model.g = model.g./repmat(sum(model.g),[model.M,1]);
        Err_g = norm(model.g(:) - gest(:))/norm(gest(:)); % added
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % added (log-likelihood calculation)
    if alg.storeLL
        Loglike = Loglike_HP_ODE(Seqs, model, alg);
        %Loglike = Loglike_Basis(Seqs, model, alg);
        model.LL(o) = Loglike;
    end
    
    % added (A, mu, g error calculation)
    if alg.storeErr
        %Err = zeros(1,3);
        %Err = zeros(1,4);
        Err = zeros(1,7);
        
        Err(1) = norm(model.A(:) - alg.truth.A(:))/norm(alg.truth.A(:));
        Err(2) = norm(model.mu(:) - alg.truth.mu(:))/norm(alg.truth.mu(:));
        model_ODE_eval = model.truth;
        model_ODE_eval.g = (model.g - model.truth.g).^2;
        model_eval = model.truth;
        model_eval.g = (model.truth.g).^2;
        Err(3) = (Kernel_Integration_Approx(model.dt, model_ODE_eval)) / ...
                 (Kernel_Integration_Approx(model.dt, model_eval));
        
        Err(4) = norm([model.mu(:); model.A(:); model.g(:)]-[alg.truth.mu(:); alg.truth.A(:); model.truth.g(:)])...
                        /norm([alg.truth.mu(:); alg.truth.A(:); model.truth.g(:)]); % combined error
        
        Err(5) = Err_A; % not official 
        Err(6) = Err_mu; % not official 
        Err(7) = Err_g; % not official 
        
        model.err(o,:) = Err;
    end
    
end

%model.LL_err = [model.LL, model.err];
%model.params = repmat([model.M, alg.alpha, alg.alphaLR, alg.alphaS, alg.alphaLR_mu], [alg.outer, 1]);
%model.params_LL_err = [model.params, model.LL_err];