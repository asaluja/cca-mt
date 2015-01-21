function regression_wrapper(file_loc, reg)
load(file_loc);
tic;
beta = linear_regression(X, Y, str2num(reg));
timeTaken = toc;
save(file_loc, 'beta');
fprintf('Time taken for normal equations: %.1f sec\n', timeTaken); 
exit

