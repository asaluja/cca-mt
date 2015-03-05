function cca_wrapper(file_locX1, file_locX2, file_locX3, file_locX4, ...
                     file_locX5, file_locX6, file_locX7, file_locX8, ...
                     file_locY1, file_locY2, file_locY3, file_locY4, ...
                     file_locY5, file_locY6, file_locY7, file_locY8, ...
                     k, reg)
load(file_locX1);
load(file_locX2);
load(file_locX3);
load(file_locX4);
load(file_locX5);
load(file_locX6);
load(file_locX7);
load(file_locX8);
X = [X1; X2; X3; X4; X5; X6; X7; X8];
load(file_locY1); 
load(file_locY2); 
load(file_locY3); 
load(file_locY4); 
load(file_locY5); 
load(file_locY6); 
load(file_locY7); 
load(file_locY8); 
Y = [Y1; Y2; Y3; Y4; Y5; Y6; Y7; Y8]; 
tic;
%[U, V, S] = cca_direct(X, Y, str2num(k), str2num(reg));
addpath('rand-cca')
opts = struct('lambda', str2num(reg), 'tmax', 2);
results = rcca(X', ones(1,size(X,1))./size(X,1), Y', str2num(k), opts);
timeTaken = toc;
U = results.x;
V = results.y;
S = results.sigma;
save(file_locY1, 'U', 'V', 'S');
fprintf('Time taken for CCA: %.1f sec\n', timeTaken);
exit
