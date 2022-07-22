function y = fSimulate_dNARX(model,u)

[W,V,gP,nu,ny] = deal(model.W,model.V,model.gP,model.nu,model.ny);
r = size(W,2);
x = zeros(nu+ny,1);
x(1:nu) = u(1:nu,1);
y = zeros(1,size(u,2));
g_eval = zeros(r,1);
for t=1:size(u,2)-1
    z = V'*x;
    for i=1:r
        g_eval(i) = polyval(gP(i,:),z(i));
    end
    y(t) = W*g_eval;
    
    x(1:nu) = u(1:nu,t+1);
    x(nu+2:nu+ny) = x(nu+1:nu+ny-1);
    x(nu+1) = y(t);
end

end


