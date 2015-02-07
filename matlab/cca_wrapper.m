function cca_wrapper(file_loc, k, reg)
load(file_loc);
tic;
%[U, V, S] = cca_direct(X, Y, str2num(k), str2num(reg));
addpath('rand-cca')
opts = struct('lambda', str2num(reg), 'tmax', 2);
results = rcca(X', ones(1,size(X,1))./size(X,1), Y', str2num(k), opts);
timeTaken = toc;
U = results.x;
V = results.y;
S = results.sigma;
save(file_loc, 'U', 'V', 'S');
fprintf('Time taken for CCA: %.1f sec\n', timeTaken);
exit
