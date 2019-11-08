function [X_train, X_test, y_train, y_test] = train_test_split(X, y, split_size)
%TRAIN_TEST_SPLIT Summary of this function goes here
%   Detailed explanation goes here
[row, ~] = size(X);
idx = randperm(row);

sz = size(idx, 2)*split_size;

X_train = X(idx(:, (1: sz)), :);
y_train = y(idx(:, (1: sz)), :);

X_test = X(idx(:, (sz+1: end)), :);
y_test = y(idx(:, (sz+1: end)), :);
end

