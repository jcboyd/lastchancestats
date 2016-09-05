% SUPPORTVECTORMACHINE for binary classification
classdef supportVectorMachine < handle
    properties
        X   % training data inputs
        K   % kernelised training data inputs
        y   % training data outputs
        k   % kernel function
        a   % dual variables
        b   % bias term
    end
    
    methods
        function obj = supportVectorMachine(X, y, kernel)
            obj.X = X;
            if strcmp(kernel, 'linear')
                obj.k = @(xi, xj) xi * xj';
            elseif strcmp(kernel, 'rbf')
                obj.k = @(xi, xj) exp(-(xi - xj) * (xi - xj)' / 2);
            end
            obj.kerneliseData();
            obj.y = y;
            obj.a = zeros(length(X), 1);
            obj.b = 0;
        end
        function kerneliseData(obj)
            % KERNELISEDATA creates kernelised training data inputs
            obj.K = zeros(length(obj.X));
            for i = 1:length(obj.X)
                for j = 1:length(obj.X)
                    obj.K(i, j) = obj.k(obj.X(i, :), obj.X(j, :));
                end
            end
        end
        function train(obj, C, tol, maxIters)
            % TRAIN trains model
            obj.smo(C, tol, maxIters);
        end
        function yhat = predict(obj, x)
            % PREDICT makes prediction on test data x
            sims = zeros(length(obj.X), 1)';
            for i = 1:length(obj.X);
                sims(i) = obj.k(x, obj.X(i, :));
            end
            yhat = sign(sims * diag(obj.y) * obj.a + obj.b);
        end
    end
    methods(Access = private)
        function yhat = trainPredict(obj, i)
            % PREDICT makes prediction on training observation i
            yhat = sign(obj.K(i, :) * diag(obj.y) * obj.a + obj.b);
        end
        function smo(obj, C, tol, maxIters)
            % SMO sequential minimal optimisation algorithm for training
            % This is the simplified version (without heuristics) detailed
            % in cs229.stanford.edu/materials/smo.pdf. For derivations of 
            % the formulae used, see github.com/jcboyd/lastchancestats.
            iters = 0;
            while iters < maxIters
                numChanged = 0;
                numCorrect = 0;
                for i = 1:length(obj.a)
                    yi = obj.y(i);
                    ai = obj.a(i);
                    Ei = obj.trainPredict(i) - yi;
                    if Ei == 0
                        numCorrect = numCorrect + 1;
                    end
                    if (yi * Ei < -tol && ai < C) || (yi * Ei > tol && ai > C)
                        % Pick random aj
                        indices = 1:length(obj.a);
                        j = randsample(indices(indices~=i), 1);
                        yj = obj.y(j);
                        Ej = obj.trainPredict(j) - yj;
                        aj = obj.a(j);
                        % Set bounds
                        if yi ~= yj
                            L = max(0, aj - ai);
                            H = min(C, C + aj - ai);
                        else
                            L = max(0, ai + aj - C);
                            H = min(C, ai + aj);
                        end
                        if L == H
                            continue
                        end
                        % Update aj
                        eta = 2 * obj.K(i, j) - obj.K(i, i) - obj.K(j, j);
                        obj.a(j) = aj - yj * (Ei - Ej) / eta;
                        % Clip value
                        if obj.a(j) > H
                            obj.a(j) = H;
                        elseif obj.a(j) < L
                            obj.a(j) = L;
                        end
                        % Check for change
                        if abs(obj.a(j) - aj) < tol
                            continue
                        end
                        obj.a(i) = ai + yi * yj * (aj - obj.a(j));
                        % Set bias term
                        b1 = obj.b - Ei - ...
                            yi * (ai - obj.a(i)) * obj.K(i, i) - ...
                            yj * (aj - obj.a(j)) * obj.K(i, j);
                        b2 = obj.b - Ej - ...
                            yi * (ai - obj.a(i)) * obj.K(i, j) - ...
                            yj * (aj - obj.a(j)) * obj.K(j, j);
                        if 0 < obj.a(i) && obj.a(i) < C
                            obj.b = b1;
                        elseif 0 < obj.a(j) && obj.a(j) < C
                            obj.b = b2;
                        else
                            obj.b = (b1 + b2) / 2;
                        end
                        numChanged = numChanged + 1;
                    end
                end
                if numChanged == 0
                    iters = iters + 1;
                    fprintf('Pass %i \t\t Error: %.4f\r', [iters, ...
                        1 - numCorrect / length(obj.a)]);
                else
                    iters = 0;
                end
            end
        end
    end
end
