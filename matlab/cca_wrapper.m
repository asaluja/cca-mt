function cca_wrapper(file_loc, k, reg)
load(file_loc);
tic;
%[U, V, S] = cca_direct(X, Y, str2num(k), str2num(reg));
%save(file_loc, 'U', 'V', 'S');
addpath('rand-cca')
opts = struct('lambda', reg, 'tmax', 2);
results = rcca(X', ones(size(X,2),1), Y', k, opts);
timeTaken = toc;
save(file_loc, 'results.x', 'results.y', 'results.sigma');
fprintf('Time taken for CCA: %.1f sec\n', timeTaken);
exit
