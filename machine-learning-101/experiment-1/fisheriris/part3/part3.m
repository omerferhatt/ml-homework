close all; clear all; clc;

load fisheriris

X = meas;
y = species;

X = standardize(X);

[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.8);

t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
model = fitcecoc(X_train, y_train, 'Learner', t);
y_pred = predict(model, X_test);

cm = confusionmat(y_test, y_pred);
figure('Name','Linear SVM Confusion Matrix','NumberTitle','off')
confusionchart(cm, {'Setosa', 'Versicolor', 'Virginica'});
title('Linear SVM')

score_calc(cm);
