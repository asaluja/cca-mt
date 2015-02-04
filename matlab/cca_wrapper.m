function cca_wrapper(file_loc, k, reg)
load(file_loc);
tic;
[U, V, S] = cca_direct(X, Y, str2num(k), str2num(reg));
timeTaken = toc;
save(file_loc, 'U', 'V', 'S');
fprintf('Time taken for CCA: %.1f sec\n', timeTaken);
exit
