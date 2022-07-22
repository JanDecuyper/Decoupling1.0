function F3 = f3Points_central_filter(N,Z,Dlevel)

r = size(Z,1);
F3 = zeros(N,N,r);

switch Dlevel
    case 'D1'
        for i=1:r
            [~,sortI] = sort(Z(i,:));
            
            S = zeros(N,N); % sorting matrix
            for s=1:N
                S(s,sortI(s)) = 1;
            end
            Sz = S*Z(i,:).';
            
            % Construct 3P filter
            Dz3 = zeros(N,N);
            
            % f'(x2)
            for d=2:N-1
                P = 2; % center point
                lCoeff = fLagrangePoly1D(Sz(d-1:d+1),P);
                
                Dz3(d,d-1) = lCoeff(1) ;
                Dz3(d,d) = lCoeff(2);
                Dz3(d,d+1) = lCoeff(3) ;
                
            end
            
            % Boundary points
            boundaryCoeff = fLagrangePoly1D(Sz(1:3),1);
            Dz3(1,1) = boundaryCoeff(1);
            Dz3(1,2) = boundaryCoeff(2);
            Dz3(1,3) = boundaryCoeff(3);
            
            boundaryCoeff = fLagrangePoly1D(Sz(end-2:end),3);
            Dz3(end,end-2) = boundaryCoeff(1);
            Dz3(end,end-1) = boundaryCoeff(2);
            Dz3(end,end) = boundaryCoeff(3);
            
            F3(:,:,i) = S.'*Dz3*S;
        end % END of r
        
    case 'D2'
        for i=1:r
            [~,sortI] = sort(Z(i,:));
            
            S = zeros(N,N); % sorting matrix
            for s=1:N
                S(s,sortI(s)) = 1;
            end
            Sz = S*Z(i,:).';
            
            % Construct 3P filter
            Dz3 = zeros(N,N);
            
            % f'(x2)
            for d=2:N-1
                P = 2; % center point
                lCoeff = fLagrangePoly2D(Sz(d-1:d+1),P);
                
                Dz3(d,d-1) = lCoeff(1) ;
                Dz3(d,d) = lCoeff(2);
                Dz3(d,d+1) = lCoeff(3) ;
                
            end
            
            % Boundary points
            boundaryCoeff = fLagrangePoly2D(Sz(1:3),1);
            Dz3(1,1) = boundaryCoeff(1);
            Dz3(1,2) = boundaryCoeff(2);
            Dz3(1,3) = boundaryCoeff(3);
            
            boundaryCoeff = fLagrangePoly2D(Sz(end-2:end),3);
            Dz3(end,end-2) = boundaryCoeff(1);
            Dz3(end,end-1) = boundaryCoeff(2);
            Dz3(end,end) = boundaryCoeff(3);
            
            F3(:,:,i) = S.'*Dz3*S;
        end % END of r
end
end