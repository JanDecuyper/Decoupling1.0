clear
close all

addpath('../Source_code/')

%% Decompose known function with exact solution

nI = 2;
nO = 2;

N = 1e2;
x = 3*rand(nI,N)-1.5;

fc.type = 'polynomial';
fc.options.degrees = [2,3];
fc.options.powers = fCombinations(nI,fc.options.degrees);
fc.coef = zeros(nO,length(fc.options.powers));

% 3 branch function with exact solution
r = 3;
fc.coef(1,1) =5.25 ;
fc.coef(1,2) = 0;
fc.coef(1,3) =-20.5;
fc.coef(1,4) = 29.875;
fc.coef(1,5) = 42.75;
fc.coef(1,6) = 31.5;
fc.coef(1,7) = -2;


fc.coef(2,1) =20.75;
fc.coef(2,2) =41;
fc.coef(2,3) =85;
fc.coef(2,4) = 109.375;
fc.coef(2,5) = 120.75;
fc.coef(2,6) = 88.5;
fc.coef(2,7) =93;

% Random function without knowledge of solution
r = 3;
%fc.coef = randn(2,7);

% Output coupled function
output_c = (fBuildRegressor(x.',fc.type,fc.options)*fc.coef.').';


% Exact solution r = 3 %
Vex = [1,3,0.5;2,1,3];
Wex = [3,0.5,-1;1,2,3];
gex_coef = [1,0.5,0,0; 2,1,0,0;1,3,0,0];

ksi = Vex.'*x;
gex_eval = zeros(r,N);
for i=1:r
    gex_eval(i,:) = polyval(gex_coef(i,:),ksi(i,:));
end

output_d_exact = Wex*gex_eval;

%% FTD

J = zeros(nO,nI,N);
J = fEdwdx_sineBasis(x,fc,fc.coef,1:nI);


% CPD solution
WVH = cpd(J,r);
W = WVH{1};
V = WVH{2};
H = WVH{3};
Z = V.'*x;
figure(10), for i=1:r, subplot(1,r,i), plot(Z(i,:),H(:,i),'k.'), end
title('CPD solution')

WVHG_init = {};
for r=3
%     W_init = rand(nO,r);
%     V_init = rand(nI,r);
%     H_init = rand(N,r);
%     G_init = rand(N,r);
%     
%     WVHG_init{1} = W_init;
%     WVHG_init{2} = V_init;
%     WVHG_init{3} = H_init;
%     WVHG_init{4} = G_init;

load('WVHG_init_toy.mat') % solution can depend on initialisation
    

options.maxit = 100;
options.alphaF = 1;
options.alphaJ = 1;
options.lambda = 0.0001;
options.randJump = 5;
options.Dlevel = 'D1';


[W,V,H,G,errorT3]= FTD_Zero_First_Reg_1p0(J, WVHG_init, x,output_c,options);
    
    Z = V.'*x;
    
    figure
    for i=1:r
        subplot(1,r,i)
        plot(Z(i,:),H(:,i),'k.'), hold on,
        plot(Z(i,:),G(:,i),'r.')
    end
    legend('H','G')
    
    figure
    for i=1:r
        subplot(1,r,i)
        plot(Z(i,:),H(:,i),'k.'), hold on,
    end    
    
    
    % % Polynomial basis OLS fit %
    degree = 3;
    gP = zeros(r,degree+1);
    
    % estimate on G %
    figure
    for i=1:r
        
        gP(i,:) = polyfit(Z(i,:)',G(:,i),degree);
        
        subplot(1,r,i)
        plot(sort(Z(i,:)),polyval([gP(i,:)],sort(Z(i,:))),'r-'), hold on,
        plot(Z(i,:),G(:,i),'k.'), hold on,
        plot(Z(i,:),G(:,i)-polyval([gP(i,:)],Z(i,:).'),'g.')
    end
    title(['G fit '])
     
    % re-evalute parametric g
    g = zeros(r,N);
    Z = V.'*x;
    
    for i=1:r
        g(i,:) = polyval(gP(i,:),Z(i,:));
    end
    
    output_d = W*g;
    
    rellerr1 = rms(output_c(1,:)-output_d(1,:))/rms(output_c(1,:)-mean(output_c(1,:)))
    rellerr2 = rms(output_c(2,:)-output_d(2,:))/rms(output_c(2,:)-mean(output_c(2,:)))
    
end

