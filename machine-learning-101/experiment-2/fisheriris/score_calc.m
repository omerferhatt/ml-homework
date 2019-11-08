function [accuracy, precision, f1_score, TPR, FPR, tbl] = score_calc(cm)
%SCORE_CALC Summary of this function goes here
%   Detailed explanation goes here
    class_number = size(cm, 1);
    
    total_sum = sum(sum(cm));
    
    TP = zeros(class_number, 1);
    TN = zeros(class_number, 1);
    FP = zeros(class_number, 1);
    FN = zeros(class_number, 1);
    
    accuracy = zeros(1, class_number);
    precision = zeros(1, class_number);
    f1_score = zeros(1, class_number);
    TPR = zeros(1, class_number);
    FPR = zeros(1, class_number);
    
    for i = 1: class_number
        TP(i) = cm(i, i);
        FP(i) = sum(cm(:, i)) - TP(i);
        FN(i) = sum(cm(i, :)) - TP(i);
        TN(i) = total_sum - (TP(i) + FP(i) + FN(i));
        
        accuracy(i) = (TP(i) + TN(i)) / ( TP(i) + FP(i) + TN(i) + FN(i));
        precision(i) = TP(i) / (TP(i) + FP(i));
        
        TPR(i) = TP(i) / (TP(i) + FN(i));
        FPR(i) = FP(i) / (FP(i) + TN(i));
        
        f1_score(i) = (2*TPR(i)*precision(i)) / (TPR(i) + precision(i));
        
        overall_accuracy = sum(accuracy) / class_number;
    end
    var_names = {'TPR', 'FPR', 'Precision', 'F1 Score', 'Accuracy'};
    class_names = {'Setosa'; 'Versicolor'; 'Virginica'};
    tbl = table(TPR', FPR', precision', f1_score', accuracy', 'VariableNames', var_names, 'RowNames', class_names)
    disp('Overall Accuracy: ')
    disp(overall_accuracy)
    
end

