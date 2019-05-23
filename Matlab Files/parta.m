%% 1
clear
close all
load data_ps3_2.mat;

%% set 1 - linear
svm_test(@Klinear, 1, 1000, set1_train, set1_test);
%% set 2 - polynomial
svm_test(@Kpoly, 2, 1000, set2_train, set2_test);
%% set 3 - gaussian
svm_test(@Kgaussian, 1, 1000, set3_train, set3_test);


%% 2
% set 4 - all of them
linear_error = svm_test2(@Klinear, 1, 1000, set4_train, set4_test);
poly_error = svm_test2(@Kpoly, 2, 1000, set4_train, set4_test);
gauss_error = svm_test2(@Kgaussian, 1.5, 1000, set4_train, set4_test);