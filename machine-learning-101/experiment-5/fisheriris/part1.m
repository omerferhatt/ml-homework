close all; clear all; clc;

load fisheriris

X = meas(:, [1 4]);
y = species;

[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.8);

[idx_train,C] = kmeans(X_train, 3);

hold on
plot(C(:,1),C(:,2),'kx', 'MarkerSize',15,'LineWidth',3)
hold on
[~,idx_test] = pdist2(C,X_test,'euclidean','Smallest',1);

gscatter(X_test(:,1),X_test(:,2),idx_test, 'rgb', 'o', 4)
legend