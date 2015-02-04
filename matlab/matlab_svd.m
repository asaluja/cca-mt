function matlab_svd(file_loc, k)
load(file_loc);
tic;
[U, S, V] = svds(avgOP, str2num(k));
S = diag(S)';
timeTaken = toc;
save(file_loc, 'U', 'V', 'S');
fprintf('Time taken for CCA: %.1f sec\n', timeTaken); 
exit

