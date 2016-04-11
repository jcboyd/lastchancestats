function res = sigma(val)
%SIGMA evaluates the sigmoid function
    res = 1 ./ (1 + exp(-val));
end