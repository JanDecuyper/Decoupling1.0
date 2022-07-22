function out = fEdwdx_sineBasis(contrib,zeta,E,index_x)
%FEDWDX Multiply a matrix E with the derivative w.r.t. x of a polynomial w(x,u).
%
%	Usage:
%		out = fEdwdx(contrib,pow,coeff,E,nx,n)
%
%	Description:
%		TODO
%
%	Output parameters:
%		out : n_out x n x N matrix that is the product of E and the
%		      derivative of the ... w(x,u) w.r.t. the elements in x
%		      at all samples. TODO
%
%	Input parameters:
%		contrib : (n+m) x N matrix containing N samples of the signals x
%		          and u
%       zeta : TODO
%       E : n_out x n_nx matrix
%       index_x : vector with indices indicating the positions in contrib
%                 of the variables w.r.t. which the derivatives are taken,
%                 e.g. if contrib = [x1 x2 x3 u1] and index_x = 1:3, then
%                 the derivatives are taken w.r.t. x1, x2, and x3.
%
%	Versions:
%		1.0 : May 9, 2017
%
%	Copyright (c) Vrije Universiteit Brussel – dept. ELEC
%   All rights reserved.
%   Software can be used freely for non-commercial applications only.
%   Disclaimer: This software is provided “as is” without any warranty.
%
%   See also fEdwdu_sineBasis

%--------------------------------------------------------------------------
% Version 1.0
%--------------------------------------------------------------------------
% {

[n_all,N] = size(contrib); % n_all = number of signals x and u; N = number of samples
[n_out,n_nx] = size(E); % n_out = number of rows in E; n_nx = number of monomials in w
n = length(index_x); % number of signals w.r.t. which derivatives are taken
% out = zeros(n_out,n,N); % Preallocate
dwdx = zeros(n_nx,n,N); % Preallocate
switch lower(zeta.type)
    case 'sine'
        L = repmat(zeta.options.L,[N,1]);
        for ii = 1:n_nx
            harmonics = repmat(zeta.options.harmonics(ii,:),[N,1]);
            for jj = 1:n
                index_cos = index_x(jj);
                index_sin = setdiff(1:n_all,index_cos);
                harmonics_sin = harmonics(:,index_sin);
                harmonics_cos = harmonics(:,index_cos);
                L_sin = L(:,index_sin);
                L_cos = L(:,index_cos);
                uPlusL_sin = contrib(index_sin,:).' + L_sin;
                uPlusL_cos = contrib(index_cos,:).' + L_cos;
                dwdx(ii,jj,:) = pi*harmonics_cos./(2*L_cos).*...
                    cos(pi*harmonics_cos.*uPlusL_cos./(2*L_cos))./sqrt(L_cos).*...
                    prod(sin(pi*harmonics_sin.*uPlusL_sin./(2*L_sin))./sqrt(L_sin),2);
            end
        end
    case 'polynomial'
        powers = zeta.options.powers;
        d_contrib = repmat(permute(contrib,[3,2,1]),[n_nx,1,1]); % n_nx x N x n_all
        for jj = 1:n
            d_powers = powers;
            d_powers(:,index_x(jj)) = d_powers(:,index_x(jj)) - 1; % Derivative w.r.t. variable index_x(jj) has one degree less in variable index_x(jj) than original polynomial
            d_powers(d_powers == -1) = 0; % The derivative of a constant is zero (= a constant)
            d_coeff = powers(:,index_x(jj)); % Polynomial coefficients of the derivative
            d_powers = permute(d_powers,[1,3,2]); % n_nx x 1 x n_all
            d_coeff = repmat(d_coeff,[1,N]); % n_nx x N
            d_powers = repmat(d_powers,[1,N,1]); % n_nx x N x n_all
            dwdx(:,jj,:) = d_coeff.*prod(d_contrib.^d_powers,3);
        end
    otherwise
        error('This type of basis function is not implemented')
end
% out = reshape(E*reshape(dwdx,n_nx,[]),n_out,n,N);
out = reshape(E*reshape(dwdx,n_nx,n*N),n_out,n,N);


%}