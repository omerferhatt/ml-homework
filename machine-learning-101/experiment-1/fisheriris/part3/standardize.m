function [standardized] = standardize(input)
%STANDARDIZE Summary of this function goes here
%   Detailed explanation goes here
standardized = zeros(size(input));

for i = 1:numel(input(1,:))
    mean_arr = mean(input(:, i));
    std_arr = std(input(:, i));
    for j = 1:numel(input(:, 1))
        standardized(j, i) = (input(j, i) - mean_arr) / std_arr;
    end
end

end

