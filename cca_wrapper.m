function cca_wrapper(file_loc, k, reg)
load(file_loc);
tic;
[A, B, r] = cca_direct(X, Y, str2num(k), str2num(reg));
timeTaken = toc;
save(file_loc, 'A', 'B', 'r');
fprintf('Time taken for CCA: %.3f sec\n', timeTaken);
exit
