function beta = linear_regression(X, Y, gamma)
%assume X is a relatively skinny dense matrix, and Y is sparse

p = size(X,2);
Cxx = X'*X + gamma*speye(p, p); 
resp = X'*Y;
beta = Cxx \ resp; 
