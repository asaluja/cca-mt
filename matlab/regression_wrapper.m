function regression_wrapper(file_locX1, file_locX2, file_locY, reg)
load(file_locX1);
load(file_locX2);
X = [X1; X2]; 
load(file_locY); 
tic;
beta = linear_regression(X, Y, str2num(reg));
timeTaken = toc;
save(file_locY, 'beta');
fprintf('Time taken for normal equations: %.1f sec\n', timeTaken); 
exit

