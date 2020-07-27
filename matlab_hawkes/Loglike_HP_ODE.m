function Loglike = Loglike_HP_ODE( Seqs, model, alg )
                                                        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Learning Hawkes processes with MLE and ordinary differential equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial 
Aest = model.A;        
muest = model.mu;

tic;

Loglike = 0; % negative log-likelihood, added

for c = 1:length(Seqs)
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

    %Amu = Amu + Tstop - Tstart; % commented out 

    dT = Tstop - Time;
    GK = Kernel_Integration_Approx(dT, model);

    Nc = length(Time);

    for i = 1:Nc

        ui = Event(i);



        ti = Time(i);             



        lambdai = muest(ui);



        if i>1

            tj = Time(1:i-1);
            uj = Event(1:i-1);

            dt = ti - tj;
            gij = Kernel_Approx(dt, model);
            auiuj = Aest(uj, :, ui);
            pij = auiuj .* gij;
            lambdai = lambdai + sum(pij(:));
        end

        Loglike = Loglike - log(lambdai);




    end

    Loglike = Loglike + (Tstop - Tstart).*sum(muest);
    Loglike = Loglike + sum( sum( GK.*sum(Aest(Event,:,:),3) ) );

end

Loglike = -Loglike;
        
    

