function [overall_accuracy, overall_precision, overall_f1_score, overall_TPR, overall_FPR, tbl_avg] = overall_score_calc(cm)
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
        
        
    end
    
    overall_accuracy = mean(accuracy);
    overall_f1_score = mean(f1_score);
    overall_precision = mean(precision);
    overall_TPR = mean(TPR);
    overall_FPR = mean(FPR);
    
    var_names = {'Avg. TPR', 'Avg. FPR', 'Avg. Precision', 'Avg. F1 Score', 'Avg. Accuracy'};
    tbl_avg = table(overall_TPR, overall_FPR, overall_precision, overall_f1_score, overall_accuracy, 'VariableNames', var_names, 'RowNames', {'3 Class Average'})
    
end
