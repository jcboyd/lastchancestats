classdef logisticRegression
    %LOGISTICREGRESSION class for logistic regression model
    properties
        X, y, w
    end
    methods
        function obj = logisticRegression(X, y)
            % initialise properties
            obj.X = X;
            obj.y = y;
            obj.w = zeros(size(X, 2), 1);
        end
        function NLL = cost(obj)
            % return cost at w
            NLL = sum(log(1 + exp(obj.X * obj.w))) - obj.y' * obj.X * obj.w;
        end
        function g = gradient(obj)
            % return gradient at w
            g = obj.X' * (sigma(obj.X * obj.w) - obj.y);
        end
        function H = hessian(obj)
            % return Hessian at w
            sig = sigma(obj.X * obj.w);
            H = obj.X' * diag(sig .* (1 - sig)) * obj.X;
        end
    end
end