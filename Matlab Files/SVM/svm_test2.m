function [errors] = svm_test2(kernel,param,C,train_data,test_data)

svm = svm_train(train_data,kernel,param,C);

% verify for training data
y_est = sign(svm_discrim_func(train_data.X,svm));
errors = find(y_est ~= train_data.y);


% evaluate against test data
y_est = sign(svm_discrim_func(test_data.X,svm));
errors = find(y_est ~= test_data.y);

fprintf('TEST RESULTS: %g of test examples were misclassified.\n',...
    length(errors)/length(test_data.y));

