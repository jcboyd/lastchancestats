function beta_new = gradientDescent(alpha, beta, g)
%GRADIENTDESCENT performs one step of vanilla gradient descent
    beta_new = beta - alpha * g;
end