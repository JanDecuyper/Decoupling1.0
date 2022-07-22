function [W,V,H,G,errorTot,argout]= FTD_Zero_First_Imp_1p0(tensor, WVHGinit, x,F,options)

% Inputs %
% ------ %
% tensor: To be decomposed
% WVHGinit: initialisation of the tensor factors. The dimensions of the
% factors define the number of branches r
% x: inputs to the multivariate function m x N
% F: the output to the function N x no
% options: structure with the fields
%     - maxit: maximum number of iterations
%     - alphaF: hyperparameter used as weight for the zeroth-order information
%     - alphaJ: hyperparameter used as weight for the first-order information
%     - Dlevel: either D1 or D2 -> the order of the derivative used in the penalty term
%     - randJump: the number of iterations until a linear initialisation of V is used
%
% Outputs %
% ------- %
% Tensor factors W, V, H, G
% Error computed after every ALS iteration

% Uses some functions from Tensorlab and the PNLSS toolbox.
%
% Jan Decuyper
% Vrije Universiteit Brussel
% June 28 2022


%% Initialisation
if size(x,1) > size(x,2)
    x = x';
end

if size(F,1) < size(F,2)
    F = F';
end

maxit = options.maxit;
alphaF = options.alphaF;
alphaJ = options.alphaJ;
randJump = options.randJump;
Dlevel = options.Dlevel;

errorTot = inf(maxit,1);
errorBestT3 = inf;

error_first = inf(maxit,1);
error_zero = inf(maxit,1);

sizeW = zeros(maxit,1);
sizeV = zeros(maxit,1);
sizeG = zeros(maxit,1);

[p n N]=size(tensor);

T1=tens2mat(tensor,1,[2 3]);
T2=tens2mat(tensor,2,[1 3]);
T3=tens2mat(tensor,3,[1 2]);

W = WVHGinit{1};
V = WVHGinit{2};
HC = WVHGinit{3};
G = WVHGinit{4};

r = size(G,2);

TolX = 1e-18;
TolFun = 1e-18;
maxIter = 1000;

storeW = zeros(size(W,1),size(W,2),maxit);
storeV = zeros(size(V,1),size(V,2),maxit);
storeG = zeros(size(G,1),size(G,2),maxit);

% Normalising hyperparameters
alphaF = alphaF/frob(F-mean(F,1))^2;
alphaJ = alphaJ/frob(tensor)^2;


%% Global ALS routine

for ALSit = 1:maxit
    
    % 1) update V
    % compute nonlinear LS update
    if ~mod(ALSit,randJump)
    V_lin = T2 * kr(HC,W) / ((HC.'*HC) .* (W.'*W));
    v0 = V_lin(:); % perturbate the solution using a linear update once every randJump
    else
        v0 = V(:); % initialise using the previous timestep
    end
    problem.solver = 'lsqnonlin';
    problem.objective = @(v) CostV(v,n,tensor,W,G,x,alphaJ,Dlevel);
    problem.x0 = v0;
    problem.options = optimset('Algorithm','levenberg-marquardt','MaxIter',maxIter,'TolX',TolX,'TolFun',TolFun,'display','off');
    v = lsqnonlin(problem);
    V = reshape(v,n,r);
    
    % Normalise V
    for rr=1:r
        sc = norm(V(:,rr));
        V(:,rr) = V(:,rr)./sc;        
    end
    
    % Monitor balance of terms
    disp(['Balance of factors - norm W: ' num2str(norm(W)) ' norm V: ' num2str(norm(V)) ' norm HC: ' num2str(norm(HC)) ' G: ' num2str(norm(G))])
    
    % 2) update G and H
    
    % Construct filters
    Z = V'*x;
    F3 = f3Points_central_filter(N,Z,'D1'); % HC := F3(D1)G
    FL = f3Points_left_filter(N,Z,Dlevel);
    FR = f3Points_right_filter(N,Z,Dlevel);

    % To solve: || [vec(F); vec(J3); vec(J3); vec(J33)] - [KF; K3; KL; KR] vec(G)||_F 
    
    %----%
    % KF %
    %----%
    KF = kron(W,eye(N));
    
    %----%
    % K3 %
    %----%   
    Fstring3 = [];
    for i=1:r
        Fstring3 =[Fstring3, 'F3(:,:,' num2str(i) '),'];
    end
    Fstring3(end) = [];
    eval(['blokF3 = blkdiag(' Fstring3 ');']);
    K3 = (kron(kr(V,W),eye(N)))*blokF3;

    %----%
    % KL %
    %----%   
    FstringL = [];
    for i=1:r
        FstringL =[FstringL, 'FL(:,:,' num2str(i) '),'];
    end
    FstringL(end) = [];
    eval(['blokFL = blkdiag(' FstringL ');']);
    KL = (kron(kr(V,W),eye(N)))*blokFL;
    
    %----%
    % KR %
    %----%   
    FstringR = [];
    for i=1:r
        FstringR =[FstringR, 'FR(:,:,' num2str(i) '),'];
    end
    FstringR(end) = [];
    eval(['blokFR = blkdiag(' FstringR ');']);
    KR = (kron(kr(V,W),eye(N)))*blokFR;

    
    K = [sqrt(alphaF)*KF;sqrt(alphaJ)*K3;sqrt(alphaJ)*KL;sqrt(alphaJ)*KR];
    
    % Solve for G
    vecG = lsqminnorm(K,[sqrt(alphaF)*fVec(F);sqrt(alphaJ)*fVec(T3);sqrt(alphaJ)*fVec(T3);sqrt(alphaJ)*fVec(T3)]);
    G = reshape(vecG,size(G));
    
    errorTot(ALSit) = frob([sqrt(alphaF)*fVec(F);sqrt(alphaJ)*fVec(T3);sqrt(alphaJ)*fVec(T3);sqrt(alphaJ)*fVec(T3)]-K*vecG);
    
    if alphaF == 0
        G = G-mean(G,1); % without zeroth order term G contains arbitrary DC
    end
    
    % Compute H %
    HC = zeros(N,r);
    HL = zeros(N,r);
    HR = zeros(N,r);
    for i=1:r
       G(:,i) = G(:,i)./norm(G(:,i)); % Normalise G
       HC(:,i) = F3(:,:,i)*G(:,i);
       HL(:,i) = FL(:,:,i)*G(:,i);
       HR(:,i) = FR(:,:,i)*G(:,i);
    end
    
    % 3) update W
    % LS update joint objective on function output F and Jacobian T
    K = [sqrt(alphaF)*G; sqrt(alphaJ)*kr(HC,V);sqrt(alphaJ)*kr(HL,V); sqrt(alphaJ)*kr(HR,V)];
    W = (K \ [sqrt(alphaF)*F;sqrt(alphaJ)*T1';sqrt(alphaJ)*T1';sqrt(alphaJ)*T1'])';

    % 4) report cost function value & visualise result
  
    if mod(ALSit,1)  == 0
        disp(['ALSit ' num2str(ALSit) ' err - '  ' 0: ' num2str(frob(F-(G*W'))/frob(F-mean(F,1))) ...
            ' - 1: ' num2str(frob(tensor-cpdgen({W,V,HC}))/frob(tensor))]) % ' ++++ ' num2str(frob(tensor-cpdgen({W,V,HC}))/frob(fVec(tensor)-mean(fVec(tensor))))]);
    end
    disp(['--------------------------------------'])
    error_first(ALSit)= frob(tensor-cpdgen({W,V,HC}))/frob(tensor);
    error_zero(ALSit) = frob(F-(G*W'))/frob(F-mean(F,1));
    
    if ALSit ~= 1
    % Cost + branches (ridge functions) + projection indices
    figure(1)
    subplot(3,1,1)
    plotyy([1:ALSit],[error_first(1:ALSit) error_zero(1:ALSit)],[1:ALSit],errorTot(1:ALSit))
    ylim([0 1.2*max([error_first(1:ALSit); error_zero(1:ALSit)])])
    legend('Rel. error 1','Rel. error 0','Total cost')
    
    for R=1:r
        subplot(3,r,r+R)
        plot(Z(R,:),G(:,R),'k.')
        subplot(3,r,2*r+R)
        bar(V(:,R))
        ylabel(['v_' num2str(R)])
    end
    pause(0.1)
    end
    
    % 5) store iteration results
    sizeW(ALSit) = norm(norm(W));
    sizeV(ALSit) = norm(norm(V));
    sizeG(ALSit) = norm(norm(G));
    
    storeW(:,:,ALSit) = W;
    storeV(:,:,ALSit) = V;
    storeG(:,:,ALSit) = G;
    
    
    if errorTot(ALSit) < errorBestT3
        Wbest = W;
        Vbest = V;
        Hbest = HC;
        Gbest = G;
        
        errorBestT3 = errorTot(ALSit);
    end
end % END of ALS

%% Outputs %
W = Wbest;
V = Vbest;
H = Hbest;
G = Gbest;

argout.sizeW = sizeW;
argout.sizeV = sizeV;
argout.sizeG = sizeG;
argout.storeW = storeW;
argout.storeV = storeV;
argout.storeG = storeG;
argout.error_first = error_first;
argout.error_zero = error_zero;

end % EOF

function CV = CostV(v,n,tensor,W,G,x,alphaJ,Dlevel)

r = size(W,2);
N = size(G,1);
V = reshape(v,n,r);
Z = V.'*x;

HL = zeros(N,r);
HR = zeros(N,r);
H3 = zeros(N,r);

% Construct filters
F3 = f3Points_central_filter(N,Z,'D1');
FL = f3Points_left_filter(N,Z,Dlevel);
FR = f3Points_right_filter(N,Z,Dlevel);

for i=1:r
    H3(:,i) = F3(:,:,i)*G(:,i);
    HL(:,i) = FL(:,:,i)*G(:,i);
    HR(:,i) = FR(:,:,i)*G(:,i);
end

T21 = tens2mat(tensor,2,[1 3]); % Reshape tensor along second dimension

% LM error series %
CV = [sqrt(alphaJ)*fVec(T21);sqrt(alphaJ)*fVec(T21);sqrt(alphaJ)*fVec(T21)] - [sqrt(alphaJ)*fVec(V*(kr(H3,W).'));sqrt(alphaJ)*fVec(V*(kr(HL,W).'));sqrt(alphaJ)*fVec(V*(kr(HR,W).'))]; % add regularisation term that penalises difference between the left and right derivative

end