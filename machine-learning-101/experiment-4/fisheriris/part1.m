close all; clear all; clc;

load fisheriris

X = meas(:, [1 2]);
y = species;


set(gcf, 'Name', 'ROC Curves', 'NumberTitle', 'off' ,'Position',[100 50 1400 950])


[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.8);
t = templateSVM('KernelFunction', 'linear');
model_svm = fitcecoc(X_train, y_train, 'Learner', t, 'ClassNames', {'setosa', 'versicolor', 'virginica'});
[y_pred_svm, scores_svm] = predict(model_svm, X_test);

tiledlayout(3,2)


nexttile
for i = 1:numel(unique(y))
    [pos, neg, ~, ~, optimal] = perfcurve(y_test, scores_svm(:, i), model_svm.ClassNames{i});
    plot(pos, neg, optimal(1), optimal(2), 'x')
    hold on
end
xlabel('FPR')
ylabel('TPR')
title('Linear SVM ROC Curve')
legend('Setosa', 'Setosa Optimal', 'Versicolor', 'Versicolor Optimal', 'Virginica', 'Virginica Optimal', 'Location', 'southeast')

nexttile
for i = 1:numel(unique(y))
    [pos, neg, ~, ~, optimal] = perfcurve(y_test, scores_svm(:, i), model_svm.ClassNames{i}, 'XCrit', 'reca', 'YCrit', 'prec');
    plot(pos, neg)
    hold on
end
xlabel('Recall')
ylabel('Precision')
title('Linear SVM ROC Curve')
legend('Setosa', 'Versicolor', 'Virginica', 'Location', 'southwest')



model_dtree = fitctree(X_train, y_train, 'ClassNames', {'setosa', 'versicolor', 'virginica'});
[y_pred_dtree, scores_dtree] = predict(model_dtree, X_test);

nexttile
for i = 1:numel(unique(y))
    [pos, neg, ~, ~, optimal] = perfcurve(y_test, scores_dtree(:, i), model_svm.ClassNames{i});
    plot(pos, neg, optimal(1), optimal(2), 'rx')
    hold on
end
xlabel('FPR')
ylabel('TPR')
title('Decision Tree ROC Curve')
legend('Setosa', 'Setosa Optimal', 'Versicolor', 'Versicolor Optimal', 'Virginica', 'Virginica Optimal', 'Location', 'southeast')

nexttile
for i = 1:numel(unique(y))
    [pos, neg, ~, ~, optimal] = perfcurve(y_test, scores_dtree(:, i), model_svm.ClassNames{i}, 'XCrit', 'reca', 'YCrit', 'prec');
    plot(pos, neg)
    hold on
end
xlabel('Recall')
ylabel('Precision')
title('Decision Tree ROC Curve')
legend('Setosa', 'Versicolor', 'Virginica', 'Location', 'southwest')



model_knn = fitcknn(X_train, y_train, 'ClassNames', {'setosa', 'versicolor', 'virginica'});
[y_pred_knn, scores_knn] = predict(model_knn, X_test);

nexttile
for i = 1:numel(unique(y))
    [pos, neg, ~, ~, optimal] = perfcurve(y_test, scores_knn(:, i), model_svm.ClassNames{i});
    plot(pos, neg, optimal(1), optimal(2), 'rx')
    hold on
end
xlabel('FPR')
ylabel('TPR')
title('KNN ROC Curve')
legend('Setosa', 'Setosa Optimal', 'Versicolor', 'Versicolor Optimal', 'Virginica', 'Virginica Optimal', 'Location', 'southeast')

nexttile
for i = 1:numel(unique(y))
    [pos, neg, ~, ~, optimal] = perfcurve(y_test, scores_knn(:, i), model_svm.ClassNames{i}, 'XCrit', 'reca', 'YCrit', 'prec');
    plot(pos, neg)
    hold on
end
xlabel('Recall')
ylabel('Precision')
title('KNN ROC Curve')
legend('Setosa', 'Versicolor', 'Virginica', 'Location', 'southwest')