function [ySim,yPred,xSim,xPred] = fFilter_NARX(model,u,y)

% Input:
%       - model: structure containing information on the regressors and the
%       coefficients.
%       - u: input signal
%       - y: true output signal (for prediction)
%
% Output:
%       - ySim: simulation output
%       - yPred: one-step-ahead prediction
%
% copyright: Jan Decuyper, Vrije Universiteit Brussels, 11 August 2020

if isfield(model,'degrees') && isfield(model,'coeff')
    Case = 'coupled';
elseif isfield(model,'V') && isfield(model,'W') && isfield(model,'gcoeff')
    Case = 'decoupled';
else
    error('Parts of the coupled or decoupled equation are missing')
end

if isfield(model,'T1')
    T1 = model.T1;
else
    T1 = 0;
end

NT = length(u);
u = u(fComputeIndicesTransient(T1,NT)); % Input with prepended transient samples
y = y(fComputeIndicesTransient(T1,NT)); % Output with prepended transient samples


switch Case
    case 'coupled'
        
        [na,nb,nk,degrees,theta] = deal(model.na,model.nb,model.nk,model.degrees,model.coeff); % number of delayed outputs;
        ... number of delayed inputs; input shift, nk=0 direct term present; degrees in the polynomial regressor; linear coefficients
            
    if size(theta,2) > size(theta,1)
        theta = theta.';
    end
    
    nx = na+nb;
    
    powers = fCombinations_JD(nx,degrees);
    npowers = size(powers,1);
    
    N = length(u);
    
    % Simulation %
    
    ySim = zeros(N,1);
    xSim = [];
    
    for t=max([na,nb])+1:N
        
        R = ones(1,npowers);
        
        xt = [];
        for i=1:na
            xt = [xt,ySim(t-i)];
        end
        
        %     if nu == 0
        %         xt = [xt,u(t)];
        %     end
        
        for i = nk:nk+nb-1
            xt = [xt,u(t-i)];
        end
        
        xSim = [xSim;xt];
        
        for pp=1:npowers
            for xx=1:nx
                R(pp) = R(pp)*xt(xx)^powers(pp,xx);
            end
        end
        
        ySim(t) = R*theta;
    end
    
    xSim = [zeros(max([na,nb]),size(xSim,2));xSim]; % initial zeros
    
    
    % Prediction %
    
    yPred = zeros(N,1);
    yPred(1:max([na,nb])) = y(1:max([na,nb]));
    
    xPred = [];
    
    for t=max([na,nb])+1:N
        
        R = ones(1,npowers);
        
        xt = [];
        for i=1:na
            xt = [xt,y(t-i)];
        end
        
        
        %     if nu == 0
        %         xt = [xt,u(t)];
        %     end
        
        for i = nk:nk+nb-1
            xt = [xt,u(t-i)];
        end
        
        xPred = [xPred;xt];
        
        for pp=1:npowers
            for xx=1:nx
                R(pp) = R(pp)*xt(xx)^powers(pp,xx);
            end
        end
        
        yPred(t) = R*theta;
    end
    
    xPred = [zeros(max([na,nb]),size(xPred,2));xPred]; % initial zeros
    
    case 'decoupled'
        
        [na,nb,nk,V,W,gcoeff] = deal(model.na,model.nb,model.nk,model.V,model.W,model.gcoeff); % number of delayed outputs;
        ... number of delayed inputs; input shift, nk=0 direct term present; degrees in the polynomial regressor; linear coefficients
            
    
    nx = na+nb;
    
    N = length(u);
    
    r = size(W,2);
    
    % Simulation %
    
    ySim = zeros(N,1);
    
    xSim = [];
    
    for t=max([na,nb])+1:N
        
        xt = [];
        for i=1:na
            xt = [xt,ySim(t-i)];
        end
        
        for i = nk:nk+nb-1
            xt = [xt,u(t-i)];
        end
        
        xSim = [xSim; xt];
        
        zt = V'*xt';
        
        gt = zeros(r,1);
        
        for rr=1:r
            gt(rr) = polyval(gcoeff(rr,:),zt(rr));
        end
        
        ySim(t) = W*gt;
    end
    
    xSim = [zeros(max([na,nb]),size(xSim,2));xSim]; % initial zeros
    
    
    % Prediction %
    
    yPred = zeros(N,1);
    yPred(1:max([na,nb])) = y(1:max([na,nb]));
    
    xPred = [];
    
    for t=max([na,nb])+1:N
        
        xt = [];
        for i=1:na
            xt = [xt,y(t-i)];
        end
        
        for i = nk:nk+nb-1
            xt = [xt,u(t-i)];
        end
        
        xPred = [xPred;xt];
        
        zt = V'*xt';
        
        gt = zeros(r,1);
        
        for rr=1:r
            gt(rr) = polyval(gcoeff(rr,:),zt(rr));
        end
        
        yPred(t) = W*gt;
    end
    
    xPred = [zeros(max([na,nb]),size(xPred,2));xPred]; % initial zeros
    
    
end

% Transient handling
if T1 ~= 0 % If there were prepended transient samples
    
    ySim = ySim(fComputeIndicesTransientRemoval(T1,NT,1)); 
    yPred = yPred(fComputeIndicesTransientRemoval(T1,NT,1)); 
    xSim = xSim(fComputeIndicesTransientRemoval(T1,NT,1),:);
    xPred = xPred(fComputeIndicesTransientRemoval(T1,NT,1),:);
end

end