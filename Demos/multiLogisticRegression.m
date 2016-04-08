classdef multiLogisticRegression
    %MULTILOGISTICREGRESSION class for multinomial logistic regression model
    properties
        X, y, W
    end
    methods
        function obj = multiLogisticRegression(X, y)
            % initialise properties
            obj.X = X;
            obj.y = y;
            obj.W = zeros(size(X, 2), size(y, 2));
        end
        function w = getWeights(obj)
            % return model weights in vector form
            w = reshape(obj.W, [size(obj.X, 2) * size(obj.y, 2), 1]);
        end
        function NLL = cost(obj)
            % return cost at w
            NLL = 0;
            for i = 1 : size(obj.X, 1)
                NLL = NLL + log(sum(exp(obj.X(i, :) * obj.W))) - obj.y(i, :) * obj.W' * obj.X(i, :)';
            end
        end
        function g = gradient(obj)
            % return gradient at w
            g = zeros(size(obj.X, 2) * size(obj.y, 2), 1);
            for i = 1 : size(obj.X, 1)
                sig = (exp(obj.X(i, :) * obj.W) / sum(exp(obj.X(i, :) * obj.W)))';
                g = g + kron(sig - obj.y(i, :)', obj.X(i, :)');
            end
        end
        function H = hessian(obj)
            % return Hessian at w
            H = zeros(size(obj.X, 2) * size(obj.y, 2));
            for i = 1 : size(obj.X, 1)
                sig = (exp(obj.X(i, :) * obj.W) / sum(exp(obj.X(i, :) * obj.W)))';
                H = H + kron(diag(sig) - sig * sig', obj.X(i, :)' * obj.X(i, :));
            end
        end
    end
end