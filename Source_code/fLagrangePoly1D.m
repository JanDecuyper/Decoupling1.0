function [l] = fLagrangePoly1D(x,P)

% Computes the coefficients of the first order derivative of a Lagrange polynomial going
% through all x: L(x) = sum_j=1^k y_k l_k(x)
% x: vector containing unequidistant absis points
% P: position index in which point the derivative is to be calculated.

k = length(x);
l = zeros(k,1);


X = x(P);

for j=1:k
    for i=1:k
        if i~=j
            Prod = 1;
            for m = 1:k
                if m~=i && m~=j
                Prod = Prod*(X-x(m))/(x(j)-x(m));
                end
            end
        l(j) = l(j) + 1/(x(j)-x(i))*Prod;
        end
    end
end

% check %
% h1 = x(2)-x(1);
% h2 = x(3)-x(2);
% lx1_y1 = -(2*h1+h2)/(h1*(h1+h2)); 
% lx2_y2 = -(h1-h2)/(h1*h2);
end