function K = fBuildRegressor(u,type,options)

[N,nu] = size(u); % Number of samples and number of inputs
switch lower(type)
    case 'sine' % See paper Andreas Svensson
        % Options
        if ~isfield(options,'nbrHarmonics')
            nbrHarmonics = 1;
        else
            nbrHarmonics = options.nbrHarmonics;
        end
        if isscalar(nbrHarmonics)
            nbrHarmonics = repmat(nbrHarmonics,[1,nu]);
        end
        if ~isfield(options,'L')
            L = max(abs(u(:)));
        else
            L = options.L;
        end
        if isscalar(L)
            L = repmat(L,[1,nu]);
        end
        
        % Create regressor
        nTerms = prod(nbrHarmonics); % Number of regressors
        harmonics = ones(1,nu); % Vector indicating which harmonic is taken in each input
        K = zeros(N,nTerms); % Preallocate regressor matrix
        uPlusL = u + repmat(L,[N,1]);
        LMatrix = repmat(L,[N,1]);
        for ii = 1:nTerms
            K(:,ii) = prod(sin(pi*repmat(harmonics,[N,1]).*uPlusL./(2*LMatrix))./sqrt(LMatrix),2); % Regressor ii
            if ii == nTerms
                break
            end
            jj = nu; % Index indicating which input harmonic to increase
            while harmonics(jj) == nbrHarmonics(jj) % Try to increase last harmonic, but if this is not possible ...
                jj = jj - 1; % ... look if the previous one can be increased
            end
            harmonics(jj) = harmonics(jj) + 1; % Increase the harmonic that could be increased
            harmonics(jj+1:end) = 1; % Put all the harmonics that could not be increased back to 1
        end
    case 'polynomial'
        % Options
        if ~isfield(options,'degrees')
            if ~isfield(options,'powers')
                degrees = 2;
            else
                degrees = unique(sum(options.powers,2));
            end
        else
            degrees = options.degrees;
        end
        if ~isfield(options,'powers')
            powers = lfPowers(nu,degrees);
        else
            powers = options.powers;
        end
        
        % Create regressor
        nTerms = size(powers,1); % Number of regressors
        K = ones(N,nTerms); % Preallocate regressor matrix
        for ii = 1:nTerms
            for jj = 1:nu
                K(:,ii) = K(:,ii).*(u(:,jj).^powers(ii,jj));
            end
        end
    otherwise
        error('This type of expansion is not implemented')
end

end

function powers = lfPowers(nu,degrees)
    % Determine total number of terms
    nTerms = 0;
    for ii = 1:length(degrees)
        nTerms = nTerms + nchoosek(nu+degrees(ii)-1,degrees(ii));
    end
    
    % List the exponents of each input in all monomials
    powers = zeros(nTerms,nu); % Preallocate the list of exponents
    index = 0; % Running index indicating the last used term
    for ii = 1:length(degrees)
        powers_ii = lfPowersHomogeneous(nu,degrees(ii)); % List of exponents in homogeneous polynomial of degree degrees(ii)
        powers(index + (1:size(powers_ii,1)),:) = powers_ii; % Insert that list in the bigger list
        index = index + size(powers_ii,1); % Update running index
    end
end

function powersHomogeneous = lfPowersHomogeneous(nu,degree)
    % Monomial representation, e.g. [1 1 2] represents u1*u1*u2
    nTerms = nchoosek(nu+degree-1,degree); % Number of terms in a homogeneous polynomial of degree degree
    monomials = ones(nTerms,degree); % Preallocating, and start from all ones => e.g. u1*u1*u1
    for ii = 2:nTerms
        monomials(ii,:) = monomials(ii-1,:); % Copy previous monomial
        jj = degree; % Index indicating which factor to change
        while monomials(ii,jj) == nu % Try to increase the last factor, but if this is not possible ...
            jj = jj - 1; % ... look the previous one that can be increased
        end
        monomials(ii,jj) = monomials(ii,jj) + 1; % Increase factor jj w.r.t. previous monomial, e.g. u1*u1*u1 -> u1*u1*u2
        monomials(ii,jj+1:degree) = monomials(ii,jj); % Monomial after u1*u1*unu is u1*u2*u2, and not u1*u2*unu
    end
    
    % Exponents representation, e.g. [2 1] represents u1^2*u2 = u1*u1*u2
    powersHomogeneous = zeros(nTerms,nu); % Preallocating
    for ii = 1:nTerms
        for jj = 1:nu
            powersHomogeneous(ii,jj) = sum(monomials(ii,:) == jj); % Count the number of appearances of input j in monomial i
        end
    end
end