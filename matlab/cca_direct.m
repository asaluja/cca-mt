function [U, V, varargout] = cca_direct(X, Y, rank, gamma)

nout = max(nargout,1)-2; %-2 because A and B are required outputs

n = size(X,1);
p1 = size(X,2);
p2 = size(Y,2);
Cxx = X'*X;
Cyy = Y'*Y;
Cxy = X'*Y;
if issparse(X) %regularizer: depends on sparse/dense
    Cxx = Cxx + gamma.*speye(p1, p1);
    Cyy = Cyy + gamma.*speye(p2, p2);
else
    Cxx = Cxx + gamma.*eye(p1, p1);
    Cyy = Cyy + gamma.*eye(p2, p2);
end

%below code unsparsifies the matrix, but Cxx and Cyy are relatively smaller
%than X,Y so this may be preferred
Cxx_root = sqrtm(full(Cxx)); 
Cyy_root = sqrtm(full(Cyy)); 
Cxy_til = (Cxx_root \ Cxy) / Cyy_root; %not a symmetric matrix
[U_cca, S_cca, V_cca] = svds(Cxy_til, rank); 
%U_correct = U_cca*sqrt(n-1);
%V_correct = V_cca*sqrt(n-1); 
U = Cxx_root \ U_cca;
V = Cyy_root \ V_cca;

if nout > 0
    varargout{1} = diag(S_cca)';
end
if nout > 1
    u = X*U;   
    v = Y*V; 
    varargout{2} = u;
    varargout{3} = v;
end
