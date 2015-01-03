function matlab_svd(file_loc, k)
load(file_loc);
tic;
[A, B, r] = cca_direct(left, right, str2num(k));
timeTaken = toc;
save(file_loc, 'A', 'B', 'r');
fprintf('Time taken for CCA: %.3f sec\n', timeTaken);
exit

