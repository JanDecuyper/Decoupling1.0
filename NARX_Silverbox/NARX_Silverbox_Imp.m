clear
close all

addpath('../Source_code/')
addpath('NN_NARX_functions/')
%% Load data
load SNLS80mV.mat
V1=V1-mean(V1); % Remove offset errors on the input measurements (these are visible in the zero sections of the input)
% The input is designed to have zero mean
%V2=V2-mean(V2); % Approximately remove the offset errors on the output measurements.
% This is an approximation because the silverbox can create itself also a small DC level

load ProcessParam.mat   % information on the experimental data

% This is an experiment consisting of concatenated sub-experiments
%  - a growing noise excitation
%  - 10 realizations of an odd random phase multisine

% Plot the input signal in the time domain
figure
t=length(V1);t=[0:t-1]/fs;
plot(t,V1,'b')
axis([0 215 -0.3 0.3])
xlabel('Time (s)')
ylabel('Amplitude (V)')
title('Composed Input signal: arrow (Growing noise) + tail (10 random phase multisines)')
shg

% process part the random phase multisines
Nstart1=Nzeros;   % parameters to extract the sub-records, Nzeros: length of zero segments
Nstart3=Nstart1+Nnoise;

uTest = V1(Nzeros+1:Nzeros+Nnoise)'; uTest = uTest-mean(uTest); % arrowhead data
yTest = V2(Nzeros+1:Nzeros+Nnoise)'; yTest = yTest-mean(yTest);

u=V1(Nstart3+[1:N1*(Nzeros+Tw+Np2)]);u=u-mean(u); % extract the N1=10 realisations + the zero segments + the transient parts
y=V2(Nstart3+[1:N1*(Nzeros+Tw+Np2)]);y=y-mean(y);

n=Nzeros+Tw+Np2; % the length of one realisation + its zero segment + its transient

% uP,Yp, periodic part
% uAll,yAll: full records, including zero start, 10 realizations of an odd random multisine

uAll=reshape(u,n,N1);
uP=uAll(Nzeros+fix(Tw*0.9)+[1:Np2],:); %eliminate zeros + transients
yAll=reshape(y,n,N1);
yP=yAll(Nzeros+fix(Tw*0.9)+[1:Np2],:); %eliminate zeros + transients


%% Train NN NARX

nu = 2;
ny = 3;

% Training data
uTrain = [];
yTrain = [];

for i=1:9
    uRTemp = uP(:,i);
    yRTemp = yP(:,i);
    
    uShift = toeplitz(uRTemp,[uRTemp(1) zeros(1,nu-1)]);
    yShift = toeplitz(yRTemp,[yRTemp(1) zeros(1,ny-1)]);
    
    
    uTrain = [uTrain [uShift(ny+1:end,:) yShift(ny:end-1,:)]'];
    yTrain = [yTrain yRTemp(ny+1:end)'];
end

uTrain_NN = num2cell(uTrain);
yTrain_NN = num2cell(yTrain);

Ntran = 250;
T1_remove = [];
T1(3:end) = T1(3:end) - [1:8]*ny;
for i=2:length(T1)
    T1_remove = [T1_remove;T1(i):T1(i)+Ntran];
end
Ind = 1:length(yTrain);
Ind(T1_remove) = [];

% Validation data
uVal = [];
yVal = [];

for i=10
    uRTemp = uP(:,i);
    yRTemp = yP(:,i);
    
    uShift = toeplitz(uRTemp,[uRTemp(1) zeros(1,nu-1)]);
    yShift = toeplitz(yRTemp,[yRTemp(1) zeros(1,ny-1)]);
    
    
    uVal = [uVal [uShift(ny+1:end,:) yShift(ny:end-1,:)]'];
    yVal = [yVal yRTemp(ny+1:end)'];
end


%% Train network
nx = 15;
maxit = 50;
net = fInit_regressor(size(uTrain,1),nx,maxit);

nn_net_opt = train(net,uTrain_NN,yTrain_NN);

yPred = cell2mat(sim(nn_net_opt,uTrain_NN));

figure
plot([yTrain',(yTrain-yPred)'])

% yPred2 = fPredict_NN_singleHidden(nn_net_opt,uTrain); % Sanity check

% Simulate
ySim = fSimulate_NN_singleHidden(nn_net_opt,nu,ny,uTrain);

figure
plot([yTrain(Ind)',(yTrain(Ind)-ySim(Ind))'])

rel_err_NN = rms(yTrain(Ind)-ySim(Ind))/rms(yTrain(Ind)-mean(yTrain(Ind)))
%% Decouple NN

% select operating points from training data
x = uTrain(:,1:500:end);
F = yTrain(:,1:500:end);

% resample points
% N = 100;
% MU = mean(uTrain,2)';
% SIGMA = 0.8*cov(uTrain'); % scale the Covariance matrix in an attempt to favour extrapolation
% x = mvnrnd(MU,SIGMA,N)'; % draw N samples from the multivariate normal distribution with mean vector MU, and covariance matrix SIGMA.
% F = sim(nn_net_opt,x);

J = fJac_nn(nn_net_opt,x);

store_results = [];
store_val = [];


for r=1:5
    WVHGinit = {};
    WVHGinit{1} = randn(size(J,1),r);
    WVHGinit{2} = randn(size(J,2),r);
    WVHGinit{3} = randn(size(J,3),r);
    WVHGinit{4} = randn(size(J,3),r);
    
    options.maxit = 15;
    options.alphaF = 1;
    options.alphaJ = 1/3; % Because central + left + right filter used
    options.randJump = 5;
    options.Dlevel = 'D1';
    
    [W01,V01,H01,G01,errorT301,argout01]= FTD_Zero_First_Imp_1p0(J, WVHGinit, x,F,options);
    options.alphaF = 0;
    % options.lambda = lambda/3;
    [W1,V1,H1,G1,errorT31,argout1]= FTD_Zero_First_Imp_1p0(J, WVHGinit, x,F,options);
    
    % Polyfit
    degree = 5;
    gP01 = zeros(r,degree+ 1);
    Gpoly = zeros(size(G01));
    Z = V01'*x;
    figure
    for i=1:r
        subplot(1,r,i)
        plot(Z(i,:),G01(:,i),'k.'), hold on,
        gP01(i,:) = polyfit(Z(i,:)',G01(:,i),degree);
        Gpoly(:,i) = polyval(gP01(i,:),Z(i,:)');
        plot(Z(i,:),Gpoly(:,i),'r.'), hold on,
        plot(Z(i,:),G01(:,i)-Gpoly(:,i),'g.')
    end
    title('01')
    
    Fd01 = W01*Gpoly';
    rel_err_01 = rms(F-Fd01)/rms(F-mean(F))
    
    gP1 = zeros(r,degree+ 1);
    Gpoly = zeros(size(G1));
    Z = V1'*x;
    figure
    for i=1:r
        subplot(1,r,i)
        plot(Z(i,:),G1(:,i),'k.'), hold on,
        gP1(i,:) = polyfit(Z(i,:)',G1(:,i),degree);
        Gpoly(:,i) = polyval(gP1(i,:),Z(i,:)');
        plot(Z(i,:),Gpoly(:,i),'r.'), hold on,
        plot(Z(i,:),G1(:,i)-Gpoly(:,i),'g.')
    end
    title('1')
    
    gP1(:,end) = 0; % arbitrary dc
    g_eval_noDC = zeros(r,size(x,2));
    for i=1:r
        g_eval_noDC(i,:) = polyval(gP1(i,:),Z(i,:));
    end
    
    y_dec_one_noDC = W1*g_eval_noDC;
    rel_err_1_noDC = rms(F-y_dec_one_noDC)/rms(F-mean(F))
    
    % estimate DC
    DC_diff = (F-y_dec_one_noDC)';
    DC = ones(size(x,2),1) \ DC_diff;
    
    W1DC = [W1,DC];
    g_eval_DC = [g_eval_noDC;ones(1,size(x,2))];
    
    y_dec_one_DC = W1DC*g_eval_DC;
    
    
    rel_err_1_DC = rms(F-y_dec_one_DC)/rms(F-mean(F))
    
    %% Construct decoupled state-space model First Zero
    
    modeld01 = struct();
    modeld01.nu = nu;
    modeld01.ny = ny;
    modeld01.V = V01;
    modeld01.W = W01;
    modeld01.gP = gP01;
    
    ySimd01 = fSimulate_dNARX(modeld01,uTrain);
    
    figure
    plot([yTrain(Ind)',(yTrain(Ind)-ySimd01(Ind))'])
    title('01')
    
    rel_err_NN
    
    rel_err_modeld01 = rms(yTrain(Ind)-ySimd01(Ind))/rms(yTrain(Ind)-mean(yTrain(Ind)))
    
    yVal01 = fSimulate_dNARX(modeld01,uVal);
    rel_err_modeld01_val = rms(yVal(Ntran:end)-yVal01(Ntran:end))/rms(yVal(Ntran:end)-mean(yVal(Ntran:end)))
    
    %% Construct decoupled state-space model First
    
    modeld1 = struct();
    modeld1.nu = nu;
    modeld1.ny = ny;
    modeld1.V = [V1,zeros(size(x,1),1)];
    modeld1.W = W1DC;
    modeld1.gP = [gP1;zeros(1,degree) 1];
    
    ySimd1 = fSimulate_dNARX(modeld1,uTrain);
    
    figure
    plot([yTrain(Ind)',(yTrain(Ind)-ySimd1(Ind))'])
    title('1')
    
    rel_err_modeld1 = rms(yTrain(Ind)-ySimd1(Ind))/rms(yTrain(Ind)-mean(yTrain(Ind)))
    
    yVal1 = fSimulate_dNARX(modeld1,uVal);
    rel_err_modeld1_val = rms(yVal(Ntran:end)-yVal1(Ntran:end))/rms(yVal(Ntran:end)-mean(yVal(Ntran:end)))
    
    yValNN = fSimulate_NN_singleHidden(nn_net_opt,nu,ny,uVal);
    rel_err_NN_val = rms(yVal(Ntran:end)-yValNN(Ntran:end))/rms(yVal(Ntran:end)-mean(yVal(Ntran:end)))
    
    %% Save results
    % r rel_err_01 relerr_1_DC rel_err_modeld01 rel_err_modeld1
    % maxLipschitz
    store_results = [store_results; r  rel_err_01 rel_err_1_DC rel_err_modeld01 rel_err_modeld1 max(max(abs(H01))) max(max(abs(H1)))];
    store_val = [store_val; r rel_err_modeld01_val rel_err_modeld1_val rel_err_NN_val];
    name = ['Imp_D1_Ft_nx15_r_' num2str(r)];
    mkdir(name)
    cd(name)
    saveAllFigures('fig')
    close all
    save('results.mat')
    cd('..')
    
end
