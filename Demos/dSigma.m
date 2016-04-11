function res = dSigma(val)
%SIGMA evaluates the derivative of the sigmoid function
    res = sigma(val) .* (1 - sigma(val));
end