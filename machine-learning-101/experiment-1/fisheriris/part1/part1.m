clear all; close all; clc;

load fisheriris;
X = meas(:, [1 3]);
y = species;

t = templateSVM('KernelFunction', 'linear', 'SaveSupportVectors', true);
model = fitcecoc(X,y,'Learners', t, 'ClassNames',{'setosa','versicolor','virginica'}, 'Verbose', 1);


d_space = 0.01;
x_mesh1 = min(X(:, 1)) : d_space : max(X(:, 1));
x_mesh2 = min(X(:, 2)) : d_space : max(X(:, 2));
[XX1, XX2] = meshgrid(x_mesh1, x_mesh2);
datagrid = [XX1(:), XX2(:)];
scores = zeros(numel(model.ClassNames), numel(datagrid(:, 1)), numel(datagrid(1, :)));

for i = 1:numel(model.ClassNames)
    
    [~, gridscores] = predict(model.BinaryLearners{i}, datagrid);
    scores(i, :, :) = gridscores;
end

figure
gscatter(X(:, 1), X(:, 2), y);
hold on

contour(XX1, XX2, reshape(scores(1,:,2),size(XX1)), [0, 0], '--k')
hold on
contour(XX1, XX2, reshape(scores(3,:,2),size(XX1)), [0, 0], '--b')
hold on
gscatter(model.BinaryLearners{1}.SupportVectors(:, 1), model.BinaryLearners{1}.SupportVectors(:, 2), model.BinaryLearners{1}.SupportVectorLabels, 'k', 'x', 10);
hold on
gscatter(model.BinaryLearners{3}.SupportVectors(:, 1), model.BinaryLearners{3}.SupportVectors(:, 2), model.BinaryLearners{3}.SupportVectorLabels, 'k', 'x', 10);
hold off
legend({'Setosa', 'Versicolor', 'Virginica', 'Decision Boundary-1', 'Decision Boundary-2','Support Vector'})