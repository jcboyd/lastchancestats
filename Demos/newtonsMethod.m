function beta_new = newtonsMethod(alpha, beta, g, H)
%NEWTONSMETHOD performs one step of Newton's method
    beta_new = beta - alpha * H \ g;
end