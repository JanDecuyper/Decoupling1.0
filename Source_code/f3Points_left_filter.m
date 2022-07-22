function FL = f3Points_left_filter(N,Z,Dlevel)

r = size(Z,1);
FL = zeros(N,N,r);

switch Dlevel
    case 'D1'
        for i=1:r
            
            [~,sortI] = sort(Z(i,:));
            
            S = zeros(N,N); % sorting matrix
            for s=1:N
                S(s,sortI(s)) = 1;
            end
            Sz = S*Z(i,:).';
            
            % Construct 3P Left filter
            DzL = zeros(N,N);
            
            % f'(x2)
            for d=1:N-2
                P = 1; % compute the derivative in the center point
                lCoeff = fLagrangePoly1D(Sz(d:d+2),P); % compute the coefficients of
                % the first derivative of the Lagrange polynomial passing through all points.
                % Results in a finite difference approximation for non-equidistant spacing.
                
                DzL(d,d) = lCoeff(1) ;
                DzL(d,d+1) = lCoeff(2);
                DzL(d,d+2) = lCoeff(3);
                
            end
            
            % Boundary points
            boundaryCoeff = fLagrangePoly1D(Sz(end-2:end),2);
            DzL(end-1,end-2) = boundaryCoeff(1);
            DzL(end-1,end-1) = boundaryCoeff(2);
            DzL(end-1,end) = boundaryCoeff(3);
            
            boundaryCoeff = fLagrangePoly1D(Sz(end-2:end),3);
            DzL(end,end-2) = boundaryCoeff(1);
            DzL(end,end-1) = boundaryCoeff(2);
            DzL(end,end) = boundaryCoeff(3);
            
            
            FL(:,:,i) = S.'*DzL*S;
            %FL_scaled(:,:,i) = FL(:,:,i);%./rms(FL(:,:,i)*G(:,i));
        end % END of r
        
    case 'D2'
        for i=1:r
            
            [~,sortI] = sort(Z(i,:));
            
            S = zeros(N,N); % sorting matrix
            for s=1:N
                S(s,sortI(s)) = 1;
            end
            Sz = S*Z(i,:).';
            
            % Construct 3P Left filter
            DzL = zeros(N,N);
            
            % f'(x2)
            for d=1:N-2
                P = 1; % compute the derivative in the center point
                lCoeff = fLagrangePoly2D(Sz(d:d+2),P); % compute the coefficients of
                % the first derivative of the Lagrange polynomial passing through all points.
                % Results in a finite difference approximation for non-equidistant spacing.
                
                DzL(d,d) = lCoeff(1) ;
                DzL(d,d+1) = lCoeff(2);
                DzL(d,d+2) = lCoeff(3);
                
            end
            
            % Boundary points
            boundaryCoeff = fLagrangePoly2D(Sz(end-2:end),2);
            DzL(end-1,end-2) = boundaryCoeff(1);
            DzL(end-1,end-1) = boundaryCoeff(2);
            DzL(end-1,end) = boundaryCoeff(3);
            
            boundaryCoeff = fLagrangePoly2D(Sz(end-2:end),3);
            DzL(end,end-2) = boundaryCoeff(1);
            DzL(end,end-1) = boundaryCoeff(2);
            DzL(end,end) = boundaryCoeff(3);
            
            
            FL(:,:,i) = S.'*DzL*S;
            %FL_scaled(:,:,i) = FL(:,:,i);%./rms(FL(:,:,i)*G(:,i));
        end % END of r
end
end