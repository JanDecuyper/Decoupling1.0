function FR = f3Points_right_filter(N,Z,Dlevel)

r = size(Z,1);
FR = zeros(N,N,r);

switch Dlevel
    case 'D1'
        for i=1:r
            [~,sortI] = sort(Z(i,:));
            
            S = zeros(N,N); % sorting matrix
            for s=1:N
                S(s,sortI(s)) = 1;
            end
            Sz = S*Z(i,:).';
            
            % Construct 3P Right filter
            DzR = zeros(N,N);
            
            % f'(x2)
            for d=3:N
                P = 3; % compute the derivative in the center point
                lCoeff = fLagrangePoly1D(Sz(d-2:d),P); % compute the coefficients of
                % the first derivative of the Lagrange polynomial passing through all points.
                % Results in a finite difference approximation for non-equidistant spacing.
                
                DzR(d,d-2) = lCoeff(1) ;
                DzR(d,d-1) = lCoeff(2);
                DzR(d,d) = lCoeff(3);
                
            end
            
            % Boundary points
            boundaryCoeff = fLagrangePoly1D(Sz(1:3),2);
            DzR(2,1) = boundaryCoeff(1);
            DzR(2,2) = boundaryCoeff(2);
            DzR(2,3) = boundaryCoeff(3);
            
            boundaryCoeff = fLagrangePoly1D(Sz(1:3),1);
            DzR(1,1) = boundaryCoeff(1);
            DzR(1,2) = boundaryCoeff(2);
            DzR(1,3) = boundaryCoeff(3);
            
            FR(:,:,i) = S.'*DzR*S;
            %FR_scaled(:,:,i) = FR(:,:,i);%./rms(FR(:,:,i)*G(:,i));
            % Construct 3P Right filter
        end
        
    case 'D2'
        for i=1:r
            [~,sortI] = sort(Z(i,:));
            
            S = zeros(N,N); % sorting matrix
            for s=1:N
                S(s,sortI(s)) = 1;
            end
            Sz = S*Z(i,:).';
            
            % Construct 3P Right filter
            DzR = zeros(N,N);
            
            % f'(x2)
            for d=3:N
                P = 3; % compute the derivative in the center point
                lCoeff = fLagrangePoly2D(Sz(d-2:d),P); % compute the coefficients of
                % the first derivative of the Lagrange polynomial passing through all points.
                % Results in a finite difference approximation for non-equidistant spacing.
                
                DzR(d,d-2) = lCoeff(1) ;
                DzR(d,d-1) = lCoeff(2);
                DzR(d,d) = lCoeff(3);
                
            end
            
            % Boundary points
            boundaryCoeff = fLagrangePoly2D(Sz(1:3),2);
            DzR(2,1) = boundaryCoeff(1);
            DzR(2,2) = boundaryCoeff(2);
            DzR(2,3) = boundaryCoeff(3);
            
            boundaryCoeff = fLagrangePoly2D(Sz(1:3),1);
            DzR(1,1) = boundaryCoeff(1);
            DzR(1,2) = boundaryCoeff(2);
            DzR(1,3) = boundaryCoeff(3);
            
            FR(:,:,i) = S.'*DzR*S;
            %FR_scaled(:,:,i) = FR(:,:,i);%./rms(FR(:,:,i)*G(:,i));
            % Construct 3P Right filter
        end
end
end