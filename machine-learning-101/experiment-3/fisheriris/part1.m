close all; clear all; clc;

load fisheriris

X = meas(:, [1 2]);
y = species;

t = templateSVM('KernelFunction', 'linear');

layout = tiledlayout(1,3);
set(gcf, 'Name', 'Linear SVM - Decision Tree - KNN Accuracy/Precision/Recall Plot', 'NumberTitle', 'off' ,'Position',[100 100 1400 600])

accuracy = zeros(1, 6);
precision = zeros(1, 6);
TPR = zeros(1, 6);

for i = 5:10
    kfold_svm = fitcecoc(X, y, 'CrossVal', 'on', 'KFold', i, 'Learners', t, 'ClassNames', unique(y));
    label = kfoldPredict(kfold_svm);
    cm = confusionmat(y, label);
    disp('Linear SVM = ')
    disp(i)
    [~, overall_precision,~ , overall_TPR, ~, ~] = overall_score_calc(cm);
    losses = kfoldLoss(kfold_svm, 'Mode', 'individual');
    accuracy(i-4) = mean(1-losses);
    precision(i-4) = overall_TPR;
    TPR(i-4) = overall_TPR;
end
nexttile
plot((5:10), accuracy, 'b-*', (5:10), precision, 'k-o', (5:10), TPR, 'r-x')
xticks(5:1:10)
ylabel('Accuracy')
title('Linear SVM')
legend({'Accuracy', 'Precision', 'Recall'}, 'Location', 'southeast')
grid on



accuracy = zeros(1, 6);
precision = zeros(1, 6);
TPR = zeros(1, 6);

for i = 5:10
    kfold_dtree = fitctree(X, y, 'CrossVal', 'on', 'KFold', i, 'ClassNames', unique(y));
    label = kfoldPredict(kfold_dtree);
    cm = confusionmat(y, label);
    disp('Decision Tree = ')
    disp(i)
    [~, overall_precision,~ , overall_TPR, ~, ~] = overall_score_calc(cm);
    losses = kfoldLoss(kfold_dtree, 'Mode', 'individual');
    accuracy(i-4) = mean(1-losses);
    precision(i-4) = overall_TPR;
    TPR(i-4) = overall_TPR;
end
nexttile
plot((5:10), accuracy, 'b-*', (5:10), precision, 'k-o', (5:10), TPR, 'r-x')
xticks(5:1:10)
xlabel('Fold Count')
title('Decision Tree')
legend({'Accuracy', 'Precision', 'Recall'}, 'Location', 'southeast')
grid on

accuracy = zeros(1, 6);
precision = zeros(1, 6);
TPR = zeros(1, 6);

for i = 5:10
    kfold_knn = fitcknn(X, y, 'CrossVal', 'on', 'KFold', i, 'ClassNames', unique(y));
    label = kfoldPredict(kfold_knn);
    cm = confusionmat(y, label);
    disp('K-Nearest Neighbor = ')
    disp(i)
    [~, overall_precision,~ , overall_TPR, ~, ~] = overall_score_calc(cm);
    losses = kfoldLoss(kfold_knn, 'Mode', 'individual');
    accuracy(i-4) = mean(1-losses);
    precision(i-4) = overall_TPR;
    TPR(i-4) = overall_TPR;
end
nexttile
plot((5:10), accuracy, 'b-*', (5:10), precision, 'k-o', (5:10), TPR, 'r-x')
xticks(5:1:10)
title('K-Nearest Neighbor')
legend({'Accuracy', 'Precision', 'Recall'}, 'Location', 'southeast')
grid on
