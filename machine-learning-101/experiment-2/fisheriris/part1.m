close all; clear all; clc;

load fisheriris

X = meas;
y = species;

layout = tiledlayout(1,3);
set(gcf, 'Name', 'Confusion Matrixes', 'NumberTitle', 'off' ,'Position',[100 100 1400 600])

[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.8);
t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
model_svm = fitcecoc(X_train, y_train, 'Learner', t);
y_pred_svm = predict(model_svm, X_test);

nexttile
cm_svm = confusionmat(y_test, y_pred_svm);
disp('Linear SVM: ')
confusionchart(cm_svm, {'Setosa', 'Versicolor', 'Virginica'});
title('Linear SVM')
score_calc(cm_svm);


model_dtree = fitctree(X_train, y_train);
y_pred_dtree = predict(model_dtree, X_test);

nexttile
cm_dtree = confusionmat(y_test, y_pred_dtree);
disp('Decision Tree: ')
confusionchart(cm_dtree, {'Setosa', 'Versicolor', 'Virginica'});
title('Decision Tree')
score_calc(cm_dtree);


model_knn = fitcknn(X,y,'NumNeighbors',5);
y_pred_knn = predict(model_knn, X_test);

nexttile
cm_knn = confusionmat(y_test, y_pred_knn);
disp('K-Nearest Heighbor: ')
confusionchart(cm_knn, {'Setosa', 'Versicolor', 'Virginica'});
title('K-Nearest Heighbor')
score_calc(cm_knn);